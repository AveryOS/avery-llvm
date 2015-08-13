#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "memmask"

namespace {

/// Check whether a given alloca instruction (AI) should be put on the safe
/// stack or not. The function analyzes all uses of AI and checks whether it is
/// only accessed in a memory safe way (as decided statically).
bool IsSafeStackAlloca(const AllocaInst *AI) {
  // Go through all uses of this alloca and check whether all accesses to the
  // allocated object are statically known to be memory safe and, hence, the
  // object can be placed on the safe stack.

  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Instruction *, 8> WorkList;
  WorkList.push_back(AI);

  // A DFS search through all uses of the alloca in bitcasts/PHI/GEPs/etc.
  while (!WorkList.empty()) {
    const Instruction *V = WorkList.pop_back_val();
    for (const Use &UI : V->uses()) {
      auto I = cast<const Instruction>(UI.getUser());
      assert(V == UI.get());

      switch (I->getOpcode()) {
      case Instruction::Load:
        // Loading from a pointer is safe.
        break;
      case Instruction::VAArg:
        // "va-arg" from a pointer is safe.
        break;
      case Instruction::Store:
        if (V == I->getOperand(0))
          // Stored the pointer - conservatively assume it may be unsafe.
          return false;
        // Storing to the pointee is safe.
        break;

      case Instruction::GetElementPtr:
        if (!cast<const GetElementPtrInst>(I)->hasAllConstantIndices())
          // GEP with non-constant indices can lead to memory errors.
          // This also applies to inbounds GEPs, as the inbounds attribute
          // represents an assumption that the address is in bounds, rather than
          // an assertion that it is.
          return false;

        // We assume that GEP on static alloca with constant indices is safe,
        // otherwise a compiler would detect it and warn during compilation.

        if (!isa<const ConstantInt>(AI->getArraySize()))
          // However, if the array size itself is not constant, the access
          // might still be unsafe at runtime.
          return false;

      /* fallthrough */

      case Instruction::BitCast:
      case Instruction::IntToPtr:
      case Instruction::PHI:
      case Instruction::PtrToInt:
      case Instruction::Select:
        // The object can be safe or not, depending on how the result of the
        // instruction is used.
        if (Visited.insert(I).second)
          WorkList.push_back(cast<const Instruction>(I));
        break;

      case Instruction::Call:
      case Instruction::Invoke: {
        // FIXME: add support for memset and memcpy intrinsics.
        ImmutableCallSite CS(I);

        // LLVM 'nocapture' attribute is only set for arguments whose address
        // is not stored, passed around, or used in any other non-trivial way.
        // We assume that passing a pointer to an object as a 'nocapture'
        // argument is safe.
        // FIXME: a more precise solution would require an interprocedural
        // analysis here, which would look at all uses of an argument inside
        // the function being called.
        ImmutableCallSite::arg_iterator B = CS.arg_begin(), E = CS.arg_end();
        for (ImmutableCallSite::arg_iterator A = B; A != E; ++A)
          if (A->get() == V && !CS.doesNotCapture(A - B))
            // The parameter is not marked 'nocapture' - unsafe.
            return false;
        continue;
      }

      default:
        // The object is unsafe if it is used in any other way.
        return false;
      }
    }
  }

  // All uses of the alloca are safe, we can place it on the safe stack.
  return true;
}

Function *RecreateFunction(Function *Func, FunctionType *NewType) {
  Function *NewFunc = Function::Create(NewType, Func->getLinkage());
  //NewFunc->copyAttributesFrom(Func);
  Func->getParent()->getFunctionList().insert(Func, NewFunc);
  NewFunc->takeName(Func);
  NewFunc->getBasicBlockList().splice(NewFunc->begin(),
                                      Func->getBasicBlockList());


  AttributeSet Attrs = Func->getAttributes();
  AttributeSet FnAttrs = Attrs.getFnAttributes();

  NewFunc->addAttributes(AttributeSet::FunctionIndex, FnAttrs);
  NewFunc->addAttributes(AttributeSet::ReturnIndex, Attrs.getRetAttributes());

  // We need to recreate the attribute set, with the right indexes
  AttributeSet NewAttrs;
  unsigned NumArgs = Func->arg_size();
  for (unsigned i = 1, j = 2; i < NumArgs+1; i++, j++) {
    if (!Attrs.hasAttributes(i)) continue;
    AttributeSet ParamAttrs = Attrs.getParamAttributes(i);
    AttrBuilder AB;
    unsigned NumSlots = ParamAttrs.getNumSlots();
    for (unsigned k = 0; k < NumSlots; k++) {
      for (AttributeSet::iterator I = ParamAttrs.begin(k), E = ParamAttrs.end(k); I != E; I++) {
        AB.addAttribute(*I);
      }
    }
    NewFunc->addAttributes(j, AttributeSet::get(Func->getContext(), j, AB));
  }


  Func->replaceAllUsesWith(
      ConstantExpr::getBitCast(NewFunc,
                               Func->getFunctionType()->getPointerTo()));
  return NewFunc;
}

class AugmentArgs : public ModulePass {
 public:
  explicit AugmentArgs(bool CompileKernel = false)
      : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
  static char ID;  // Pass identification, replacement for typeid
  const char *getPassName() const override { return "AugmentArgs"; }

 private:
  Function *Func;
  Type *IntPtrTy;
  const DataLayout *DL;
  void protectFunction(Function &F, Value *Mask);
  bool safePtr(Value *I, DenseSet<Value *> &prot, SmallSet<Value *, 8> &phis, int64_t offset, int64_t size, Value *&target, bool CanOffset);
  void protectValueAndSeg(Function &F, DenseSet<Value *> &prot, Instruction *I, unsigned PtrOp, Value *Mask);
  void protectValue(Function &F, DenseSet<Value *> &prot, Value *Ptr, Value *Mask, bool CanOffset);
};

struct Info {
    Instruction *i;
    unsigned ptr;
};

const int64_t GuardSize = 0x1000;

bool AugmentArgs::safePtr(Value *I, DenseSet<Value *> &prot, SmallSet<Value *, 8> &phis, int64_t offset, int64_t size, Value *&target, bool CanOffset) {
  target = I;

  if (prot.count(I)) {
    return true;
  } else if (auto GEP = dyn_cast<GetElementPtrInst>(I)) {
      if (!CanOffset) {
        return false;
      }
      APInt aoffset(DL->getPointerSizeInBits(), 0);
      if (!GEP->accumulateConstantOffset(*DL, aoffset)) {
          return false;
      }
      offset += aoffset.getSExtValue();

      if (offset + size >= GuardSize || offset <= -GuardSize) {
          return false;
      }

      if (safePtr(GEP->getPointerOperand(), prot, phis, offset, size, target, CanOffset)) {
          prot.insert(I);
          return true;
      }
  } else if (auto BC = dyn_cast<BitCastInst>(I)) {
      auto Src = BC->getSrcTy();
      if (Src->isPointerTy()) {
        if (safePtr(BC->getOperand(0), prot, phis, offset, size, target, CanOffset)) {
            return true;
        }
      }
  } else if (auto PN = dyn_cast<PHINode>(I)) {
      if (!phis.insert(PN).second) {
        return false;
      }
      unsigned SafeVals = 0;
      for (Value *val: PN->incoming_values()) {
        Value *PTarget;
        if (safePtr(val, prot, phis, offset, size, PTarget, CanOffset)) {
            SafeVals++;
        }
      }
      if (PN->getNumIncomingValues() == SafeVals) {
        llvm::errs() << "OPT PHI REF in " << Func->getName() << " ";
          I->dump();
        llvm::errs() << "\n";
        prot.insert(I);
        return true;
      }
      if (PN->getNumIncomingValues() - SafeVals <= 1) {
        llvm::errs() << "SAFE PHI REF in " << Func->getName() << " ";
          I->dump();
        llvm::errs() << "\n";
      }
      return false;
  } else if (auto A = dyn_cast<Argument>(I)) {
      return false;//A->getArgNo() == 1;
  } else if (dyn_cast<ConstantPointerNull>(I)) {
      return true;
  } else if (auto AI = dyn_cast<AllocaInst>(I)) { // References to the stack are safe
      /*if (IsSafeStackAlloca(AI)) {

        llvm::errs() << "ALLOCA REF in " << Func->getName() << " ";
          I->dump();
        llvm::errs() << "\n";/*FUNC\n";
          Func->dump();
        llvm::errs() << "/ALLOCA\n";
      } else {
        llvm::errs() << "UNSAFE ALLOCA REF in " << Func->getName() << " ";
          I->dump();
        llvm::errs() << "\n";

      }*/
      return true;
  } else if (isa<GlobalValue>(I)) {
      return true;
  } else if (isa<CallInst>(I)) {
       /* llvm::errs() << "SAFE CALL REF in " << Func->getName() << " ";
          I->dump();
        llvm::errs() << "\n";*/
      return true;
  }
  return false;
}

void AugmentArgs::protectValue(Function &F, DenseSet<Value *> &prot, Value *Ptr, Value *Mask, bool CanOffset) {
  IRBuilder<> IRB(F.getEntryBlock().getFirstInsertionPt());

  Value *Target;
  auto MemType = Ptr->getType()->getPointerElementType();
  SmallSet<Value *, 8> phis;

  if (safePtr(Ptr, prot, phis, 0, CanOffset ? DL->getTypeAllocSize(MemType) : 0, Target, CanOffset))
    return;

  if (auto TI = dyn_cast<Instruction>(Target)) {
    // Insert it after the instruction generating the pointer
    BasicBlock::iterator it = std::next(BasicBlock::iterator(TI));
    
    // Skip Phis
    while (dyn_cast<PHINode>(&*it)) ++it;

    IRB.SetInsertPoint(it);
  }

/*    ArrayRef<Type *> ArgType(Target->getType());
  ArrayRef<Value *> Arg(Target);
  auto Asm = InlineAsm::get(FunctionType::get(Target->getType(), ArgType, false), "andq %r15, $0", "=rm,0,~{dirflag},~{fpsr},~{flags}", false);
  auto MaskedPtr = IRB.CreateCall(Asm, Arg);*/

  auto PtrVal = IRB.CreatePtrToInt(Target, IntPtrTy);
  auto MaskedVal = IRB.CreateAnd(PtrVal, Mask); //0xBADF00D // 128GB space - 0x1FFFFFFFFF // 0xBFFFFFFFFFFFFFFF
  auto MaskedPtr = IRB.CreateIntToPtr(MaskedVal, Target->getType());

  auto UI = Target->use_begin(), E = Target->use_end(); // TODO: Pass Use& to this function and ensure it gets updated
  for (; UI != E;) {
    Use &U = *UI;
    ++UI;
    if (PtrVal == U.getUser())
      continue;
    if (isa<Constant>(U.getUser())) {
    llvm::errs() << "CONSTANT in " << F.getName() << "\n";
      U.getUser()->dump();
    llvm::errs() << "DURING PROT OF \n";
      Ptr->dump();
    llvm::errs() << "END\n";

    }
    assert(!dyn_cast<Constant>(U.getUser()));

    // Limit changes to the current function
    if (auto I = dyn_cast<Instruction>(U.getUser())) {
      if (I->getParent()->getParent() == &F) {
        U.set(MaskedPtr);

      }
    }
   /*if (auto *C = dyn_cast<Constant>(U.getUser())) {
     if (!isa<GlobalValue>(C)) {
       C->replaceUsesOfWithOnConstant(Target, MaskedPtr, &U);
       continue;
     }
   }*/
  }

  prot.insert(MaskedPtr);
}

void AugmentArgs::protectValueAndSeg(Function &F, DenseSet<Value *> &prot, Instruction *I, unsigned PtrOp, Value *Mask) {
  protectValue(F, prot, I->getOperand(PtrOp), Mask, true); // getOperandUse
  IRBuilder<> IRB(I);
/*
      auto SegPtrVal = IRB.CreatePtrToInt(I.i->getOperand(I.ptr), IntPtrTy);
      auto SegPtr = IRB.CreateIntToPtr(SegPtrVal, MemType->getPointerTo(256));
    //auto SegPtr = IRB.CreateAddrSpaceCast(I.i->getOperand(I.ptr), MemType->getPointerTo(256)); CRASHES
    I.i->setOperand(I.ptr, SegPtr); SegPtr*/

}

void AugmentArgs::protectFunction(Function &F, Value *Mask) {
  DenseSet<Value *> prot;

  for (auto &BB: F) {
    for (BasicBlock::iterator Iter = BB.begin(), E = BB.end(); Iter != E;) {
      Instruction *I = Iter++;
      if (isa<LoadInst>(I)) {
        protectValueAndSeg(F, prot, I, 0, Mask);
      } else if (isa<StoreInst>(I)) {
        protectValueAndSeg(F, prot, I, 1, Mask);
      } else if (auto AI = dyn_cast<AtomicRMWInst>(I)) {
        protectValueAndSeg(F, prot, I, AI->getPointerOperandIndex(), Mask);
      } else if (auto AI = dyn_cast<AtomicCmpXchgInst>(I)) {
        protectValueAndSeg(F, prot, I, AI->getPointerOperandIndex(), Mask);
      } else if (auto RI = dyn_cast<ReturnInst>(I)) {
          auto Arg = RI->getReturnValue(); //getArgOperandUse
          if (Arg && Arg->getType()->isPointerTy())
            protectValue(F, prot, Arg, Mask, false);
      } else if (auto CI = dyn_cast<CallInst>(I)) {
       /* for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
          auto Arg = CI->getArgOperand(i); //getArgOperandUse
          if (Arg->getType()->isPointerTy())//i == 1 && 
            protectValue(F, prot, Arg, Mask, false);
        }*/
      } else {
        if (I->mayReadOrWriteMemory()) {
          llvm::outs() << "\nWarning, touches memory! - ";
          I->print(llvm::outs());
          llvm::outs() << "\n";
        }
      }
    }
  }
}

bool AugmentArgs::runOnModule(Module &M) {
  DL = &M.getDataLayout();
  IntPtrTy = DL->getIntPtrType(M.getContext());

  for (auto Iter = M.begin(), E = M.end(); Iter != E; ) {
    Function &F = *(Iter++);
    if (F.isIntrinsic())
      continue;

/*
    llvm::errs() << "PREADDCALL" << F.getName() << "\n";
    M.dump();
    llvm::errs() << "/PREADDCALL\n";
*/

    SmallVector<Type *, 8> ArgTypes;
    ArgTypes.push_back(IntPtrTy);
    for (auto Type : F.getFunctionType()->params()) {
      ArgTypes.push_back(Type);
    }
    auto NFTy = FunctionType::get(F.getReturnType(), ArgTypes, F.isVarArg());

    auto &NF = *RecreateFunction(&F, NFTy);

    auto NewArg = NF.arg_begin();
    NewArg->setName("Mask");
    Value *Mask = &*NewArg;
    ++NewArg;
    for (auto &Arg : F.args()) {
      Arg.replaceAllUsesWith(NewArg);
      NewArg->takeName(&Arg);
      ++NewArg;
    }

    F.eraseFromParent();

    NF.setCallingConv(CallingConv::Sandbox);

    if (NF.hasExternalLinkage() && NF.getName() == "main")
      Mask = ConstantInt::get(IntPtrTy, -1);

    for (auto &BB: NF) {
      for (BasicBlock::iterator Iter = BB.begin(), E = BB.end(); Iter != E;) {
        Instruction &I = *(Iter++);

        if (auto OldCall = dyn_cast<CallInst>(&I)) {
          if (isa<IntrinsicInst>(OldCall))
            continue;

          FunctionType *CallFTy = cast<FunctionType>(
              OldCall->getCalledValue()->getType()->getPointerElementType());

          SmallVector<Type *, 8> ArgTypes;
          ArgTypes.push_back(IntPtrTy);
          for (auto Type : CallFTy->params()) {
            ArgTypes.push_back(Type);
          }
          auto NCallFTy = FunctionType::get(CallFTy->getReturnType(), ArgTypes, CallFTy->isVarArg());

          SmallVector<Value *, 8> Args;
          Args.push_back(Mask);
          for (unsigned I = 0; I < OldCall->getNumArgOperands(); ++I) {
            Value *Arg = OldCall->getArgOperand(I);
            Args.push_back(Arg);
          }

          auto CastFunc = new BitCastInst(OldCall->getCalledValue(), NCallFTy->getPointerTo(),
                                      OldCall->getName() + ".arg_cast", OldCall);
          CastFunc->setDebugLoc(OldCall->getDebugLoc());
          CallInst *NewCall = CallInst::Create(CastFunc, Args, "", OldCall);
          NewCall->setDebugLoc(OldCall->getDebugLoc());
          NewCall->takeName(OldCall);
          NewCall->setCallingConv(CallingConv::Sandbox);
          NewCall->setTailCall(OldCall->isTailCall());


          AttributeSet Attrs = OldCall->getAttributes();
          AttributeSet NAttrs = NewCall->getAttributes();
          AttributeSet FnAttrs = Attrs.getFnAttributes();
          NAttrs.addAttributes(NF.getContext(), AttributeSet::FunctionIndex, FnAttrs);
          NAttrs.addAttributes(NF.getContext(), AttributeSet::ReturnIndex, Attrs.getRetAttributes());

          // We need to recreate the attribute set, with the right indexes
          AttributeSet NewAttrs;
          for (unsigned i = 1, j = 2; i < NewCall->getNumArgOperands()+1; i++, j++) {
            if (!Attrs.hasAttributes(i)) continue;
            AttributeSet ParamAttrs = Attrs.getParamAttributes(i);
            AttrBuilder AB;
            unsigned NumSlots = ParamAttrs.getNumSlots();
            for (unsigned k = 0; k < NumSlots; k++) {
              for (AttributeSet::iterator I = ParamAttrs.begin(k), E = ParamAttrs.end(k); I != E; I++) {
                AB.addAttribute(*I);
              }
            }
            NAttrs.addAttributes(NF.getContext(), j, AttributeSet::get(NF.getContext(), j, AB));
          }

          NewCall->setAttributes(NAttrs);

          
          OldCall->replaceAllUsesWith(NewCall);
          OldCall->eraseFromParent();
        }
      }
    }
/*

    llvm::errs() << "ADDCALL" << NF.getName() << "\n";
    M.dump();
    llvm::errs() << "/ADDCALL\n";
*/
    Func = &NF;
    protectFunction(NF, Mask);
/*
    llvm::errs() << "PROTF" << NF.getName() << "\n";
    M.dump();
    llvm::errs() << "/PROTF\n";*/

  }
/*
    llvm::errs() << "DONEDUMP\n";
    M.dump();
    llvm::errs() << "/DONEDUMP\n";*/
  return true;
}


class MemMask : public FunctionPass {
  const DataLayout *DL;

  Type *StackPtrTy;
  Type *IntPtrTy;
  Type *Int32Ty;
  Type *Int8Ty;



public:
  static char ID; // Pass identification, replacement for typeid.
  MemMask() : FunctionPass(ID), DL(nullptr) {
    initializeMemMaskPass(*PassRegistry::getPassRegistry());
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    //AU.addRequired<AliasAnalysis>();
  }



  virtual bool doInitialization(Module &M) {
    DL = &M.getDataLayout();

    StackPtrTy = Type::getInt8PtrTy(M.getContext());
    IntPtrTy = DL->getIntPtrType(M.getContext());
    Int32Ty = Type::getInt32Ty(M.getContext());
    Int8Ty = Type::getInt8Ty(M.getContext());

    return false;
  }

  bool runOnFunction(Function &F);

}; // class SafeStack

bool MemMask::runOnFunction(Function &F) {
  return true;
}

} // end anonymous namespace

char MemMask::ID = 0;
INITIALIZE_PASS_BEGIN(MemMask, "mem-mask",
                      "mem-mask instrumentation pass", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(MemMask, "mem-mask", "mem-mask instrumentation pass",
                    false, false)

FunctionPass *llvm::createMemMaskPass() { return new MemMask(); }

char AugmentArgs::ID = 0;
INITIALIZE_PASS(
    AugmentArgs, "augment-args",
    "AugmentArgsPass"
    "ModulePass",
    false, false)
ModulePass *llvm::createAugmentArgsPass() {
  return new AugmentArgs();
}
