#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
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
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "memmask"

namespace {

Function *RecreateFunction(Function *Func, FunctionType *NewType) {
  Function *NewFunc = Function::Create(NewType, Func->getLinkage());
  //NewFunc->copyAttributesFrom(Func);
  Func->getParent()->getFunctionList().insert(Func, NewFunc);
  NewFunc->takeName(Func);
  NewFunc->getBasicBlockList().splice(NewFunc->begin(),
                                      Func->getBasicBlockList());

  if (Func->hasPersonalityFn()) {
    NewFunc->setPersonalityFn(Func->getPersonalityFn());
  }

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
  void protectValue(Function &F, DenseSet<Value *> &prot, Use &PtrUse, Value *Mask, bool CanOffset);
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
    Value *UnsafeRef = nullptr;
    for (Value *val: PN->incoming_values()) {
      Value *PTarget;
      if (safePtr(val, prot, phis, offset, size, PTarget, CanOffset)) {
          SafeVals++;
      } else {
        UnsafeRef = PTarget;
      }
    }
    if (PN->getNumIncomingValues() == SafeVals) {
      prot.insert(I);
      return true;
    }
    if (PN->getNumIncomingValues() - SafeVals <= 1) {
      target = UnsafeRef; // COMMENT OUT TO NOT FOLLOW PHIS
      return false;
    }
    return false;
  } else if (dyn_cast<Argument>(I)) {
    return false;//A->getArgNo() == 1;
  } else if (dyn_cast<ConstantPointerNull>(I)) {
    return true;
  } else if (dyn_cast<AllocaInst>(I)) { // References to the stack are safe
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
  } else if (isa<InvokeInst>(I)) {
    // We cannot insert protection instructions after an InvokeInst,
    // so we require that results are already protected.
    return true;
  } else if (isa<CallInst>(I)) {
    return true;
  }
  return false;
}
/*
void UpdateConstantUse(Function &F, Constant *C, Value *From, Value *To) {
  for (Use &U : C->uses()) {
    Constant *UC = dyn_cast<Constant>(U.getUser());

    if (!UC) {
      auto I = dyn_cast<Instruction>(CU->getUser());
      assert(I);

      if (I->getParent()->getParent() == &F) {
        C->handleOperandChange(From, To, CU); // Need a copy of the Constant
      }
    }

    if (isa<GlobalValue>(UC))
      return;

    UpdateConstantUse(D, UC, From, To);
  }
}
*/
void AugmentArgs::protectValue(Function &F, DenseSet<Value *> &prot, Use &PtrUse, Value *Mask, bool CanOffset) {
  IRBuilder<> IRB(F.getEntryBlock().getFirstInsertionPt());

  Value *Ptr = PtrUse.get();
  Value *Target;
  auto MemType = Ptr->getType()->getPointerElementType();
  SmallSet<Value *, 8> phis;

  if (safePtr(Ptr, prot, phis, 0, CanOffset ? DL->getTypeAllocSize(MemType) : 0, Target, CanOffset))
    return;

  // TODO: If Target is a Constant, call safeConstantPtr which has a map of Constants to protected values
  // If a safe one is found, recreate the instructions required to get the constant,
  //    <- this requires runtime instructions, another AND might be just as fast
  //    <- bitcasts would be free
  // Add Target -> MaskedPtr to the constant map

  if (auto TI = dyn_cast<Instruction>(Target)) {
    // Insert it after the instruction generating the pointer
    BasicBlock::iterator it = std::next(BasicBlock::iterator(TI));
    
    // Skip Phis
    while (dyn_cast<PHINode>(&*it)) ++it;

    IRB.SetInsertPoint(it);
  }

  auto PtrVal = IRB.CreatePtrToInt(Target, IntPtrTy);
  auto MaskedVal = IRB.CreateAnd(PtrVal, Mask);
  auto MaskedPtr = IRB.CreateIntToPtr(MaskedVal, Target->getType(), "P_" + Target->getName());

  auto UI = Target->use_begin(), E = Target->use_end(); // TODO: Pass Use& to this function and ensure it gets updated
  for (; UI != E;) {
    Use &U = *UI;
    ++UI;
    if (PtrVal == U.getUser())
      continue;

    // We can't handle constants. Users can be outside the function.
    // safePtr cannot look into constant, since we can't replace uses of them.
    if (auto C = dyn_cast<Constant>(U.getUser())) {
      llvm::errs() << "CONSTANT in " << F.getName() << "\n";
        C->dump();
      llvm::errs() << "DURING PROT OF \n";
        Ptr->dump();
      llvm::errs() << "END\n";
/*
      auto CU = ConstantUse(C);
      assert(CU);
      auto I = dyn_cast<Instruction>(CU->getUser());
      assert(I);

      if (I->getParent()->getParent() == &F) {
        C->handleOperandChange(Target, MaskedPtr, CU);
      }*/
      continue;
    } else if (auto I = dyn_cast<Instruction>(U.getUser())) {
      if (I->getParent()->getParent() == &F) {
        U.set(MaskedPtr);
      }
    } else {
      llvm::errs() << "UNKNOWN USER in " << F.getName() << "\n";
        U.getUser()->dump();
      llvm::errs() << "DURING PROT OF \n";
        Ptr->dump();
      llvm::errs() << "END\n";
      assert(0);
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
  protectValue(F, prot, I->getOperandUse(PtrOp), Mask, true);
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
          auto Arg = RI->getReturnValue();
          if (Arg && Arg->getType()->isPointerTy())
            protectValue(F, prot, RI->getOperandUse(0), Mask, false);
      } else if (dyn_cast<InvokeInst>(I)) {
      } else if (dyn_cast<CallInst>(I)) {
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

        if (auto OldCall = dyn_cast<InvokeInst>(&I)) {
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
          InvokeInst *NewCall = InvokeInst::Create(CastFunc, OldCall->getNormalDest(), OldCall->getUnwindDest(), Args, "", OldCall);
          NewCall->setDebugLoc(OldCall->getDebugLoc());
          NewCall->takeName(OldCall);
          NewCall->setCallingConv(CallingConv::Sandbox);


          AttributeSet Attrs = OldCall->getAttributes();
          AttributeSet NAttrs = NewCall->getAttributes();
          AttributeSet FnAttrs = Attrs.getFnAttributes();
          NAttrs.addAttributes(NF.getContext(), AttributeSet::FunctionIndex, FnAttrs);
          NAttrs.addAttributes(NF.getContext(), AttributeSet::ReturnIndex, Attrs.getRetAttributes());

          // We need to recreate the attribute set, with the right indexes
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
        } else if (auto OldCall = dyn_cast<CallInst>(&I)) {
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

struct Range {
  bool Unknown;
  int64_t RelStart;
  int64_t RelEnd;

  static Range bottom() {
    Range r;
    r.RelStart = INT64_MIN;
    r.RelEnd = INT64_MAX;
    r.Unknown = false;
    return r;
  }

  static Range unknown() {
    Range r;
    r.Unknown = true;
    return r;
  }

  static Range exact() {
    Range r;
    r.Unknown = false;
    r.RelStart = 0;
    r.RelEnd = 0;
    return r;
  }

  static std::string num(int64_t n) {
    std::stringstream s;
    if (n == INT64_MIN) {
      s << "-∞";
    } else if (n == INT64_MAX) {
      s << "+∞";
    } else {
      s << n;
    }
    return s.str();
  }

  bool allowed() {
    if (Unknown) {
      return false;
    } else {
      return (RelStart > -512 && RelStart <= 0) && (RelEnd < 512 && RelEnd >= 0);
    }
  }

  std::string str() {
    if (Unknown) {
      return "⊥";
    } else {
      std::stringstream s;
      s << "[" << num(RelStart) << ", " << num(RelEnd) << "]";
      return s.str();
    }
  }

  static int64_t offset(int64_t base, int64_t offset) {
    // base - offset where base is RelStart should return -INF if oob
    // base + offset where base is RelEnd should return INF if oob

    if (base == INT64_MIN || base == INT64_MAX) {
      return base;
    }

    const int64_t bound = 20000;

    if (offset > bound) {
      return INT64_MAX;
    }
    if (offset < -bound) {
      return INT64_MIN;
    }

    int64_t n = base + offset;

    if (n < -bound) {
      return INT64_MIN;
    }

    if (n > bound) {
      return INT64_MAX;
    }
    if (n < -bound) {
      return INT64_MIN;
    }

    return n;
  }

  Range offset(int64_t o) {
    Range r = *this;

    r.RelStart = offset(r.RelStart, o);
    r.RelEnd = offset(r.RelEnd, o);
    return r;
  }

  Range widen(Range old) {
    Range r = old;

    if (old.Unknown) {
      return *this;
    }

    if (RelStart < old.RelStart) {
      r.RelStart = INT64_MIN;
    }

    if (RelEnd > old.RelEnd) {
      r.RelEnd = INT64_MAX;
    }

    return r;
  }
};

bool operator!=(const Range& lhs, const Range& rhs) {
  if (lhs.Unknown != rhs.Unknown)
    return true;
  if (lhs.Unknown)
    return false;
  return lhs.RelStart != rhs.RelStart || lhs.RelEnd != rhs.RelEnd;
}

class MemMask : public FunctionPass {
  const DataLayout *DL;

  Type *StackPtrTy;
  Type *IntPtrTy;
  Type *Int32Ty;
  Type *Int8Ty;
  Value *Mask;


public:
  typedef DenseMap<Value *, Range> State;

  static char ID; // Pass identification, replacement for typeid.
  MemMask() : FunctionPass(ID), DL(nullptr) {
    initializeMemMaskPass(*PassRegistry::getPassRegistry());
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
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

private:
  State ExecuteI(const State &InState, Instruction *I);
  State ExecuteB(const State &InState, BasicBlock *BB);
  State Join(const State &A, State &B);


}; // class SafeStack

Range GetRange(const MemMask::State &S, Value *V) {
   auto it = S.find(V);
   if (it == S.end()) {
     return Range::bottom();
   } else {
     return (*it).getSecond();
   }
}

Range JoinRange(const Range &A, const Range &B) {
  if (A.Unknown) {
    return B;
  }
  if (B.Unknown) {
    return A;
  }
  Range r = A;
  r.RelStart = std::min(A.RelStart, B.RelStart);
  r.RelEnd = std::max(A.RelEnd, B.RelEnd);
  return r;
}

MemMask::State MemMask::ExecuteI(const State &InState, Instruction *I) {
  State R = InState;

  if (auto C = dyn_cast<CastInst>(I)) {
    if (C->isNoopCast(*DL)) {
      R[C] = R[C->getOperand(0)];
    }
  } else if (auto C = dyn_cast<BinaryOperator>(I)) {
    if (C->getOperand(1) == Mask) { // AND with [0, 0] is always valid. A special MASK value is not needed
      R[C] = Range::exact();
    }
  } else if (auto GEP = dyn_cast<GetElementPtrInst>(I)) {
    APInt aoffset(DL->getPointerSizeInBits(), 0);
    if (GEP->accumulateConstantOffset(*DL, aoffset)) {
      auto PtrR = GetRange(R, GEP->getPointerOperand());
      int64_t offset = aoffset.getSExtValue();
      R[GEP] = PtrR.offset(offset);
    }
  } else if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
    unsigned Ptr = isa<LoadInst>(I) ? 0 : 1;
    auto Val = I->getOperand(Ptr);
    auto MemType = Val->getType()->getPointerElementType();
    Range r = Range::exact(); ;
    r.RelEnd -= DL->getTypeAllocSize(MemType) - 1;
    //R[Val] = r;
  } else if (auto PN = dyn_cast<PHINode>(I)) {
    for (Value *val: PN->incoming_values()) {
      R[PN] = JoinRange(R[PN], GetRange(R, val));
    }
  }

  return R;
}

MemMask::State MemMask::ExecuteB(const State &InState, BasicBlock *BB) {
  State r = InState;
  for (auto &I: *BB) {
    r = ExecuteI(r, &I);
  }
  return r;
}

MemMask::State MemMask::Join(const MemMask::State &A, MemMask::State &B) {
  State R = A;
  for (auto &v: R) {
    v.getSecond() = JoinRange(v.getSecond(), B[v.getFirst()]);
  }
  return R;
}

bool Diff(MemMask::State &A, MemMask::State &B) {
  for (auto &v: A) {
    if (v.getSecond() != B[v.getFirst()])
      return true;
  }
  return false;
}

void Dump(MemMask::State S) {
  for (auto &v: S) {
    if (v.getFirst()->hasName()) {
      llvm::errs() << "  %" << v.getFirst()->getName() << " has value " << v.getSecond().str() << "\n";
    } else {
      v.getFirst()->dump();
      llvm::errs() << "    has value " << v.getSecond().str() << "\n";
    }
  }
}

bool MemMask::runOnFunction(Function &F) {
  Mask = &*F.arg_begin();
  //LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  State unknown;

  bool debug = false;

  for (auto &A: F.args()) {
    unknown.insert(std::pair<Value *, Range>(&A, Range::exact()));
  }


  for (auto &BB: F) {
    for (auto &I: BB) {
      if (!I.getType()->isVoidTy())
        unknown.insert(std::pair<Value *, Range>(&I, Range::unknown()));
    }
  }

  std::vector<BasicBlock *> WorkList;

  DenseMap<BasicBlock *, State> InMap;
  DenseMap<BasicBlock *, State> OutMap;

  for (auto &BB: F) {
    WorkList.insert(WorkList.begin(), &BB);
    InMap.insert(std::pair<BasicBlock *, State>(&BB, unknown));
    OutMap.insert(std::pair<BasicBlock *, State>(&BB, unknown));
  }

  //WorkList.push_back(&F.getEntryBlock());

  if (debug)
    llvm::errs() << "DURING ANALYSIS OF " << F.getName() << "\n";

  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.back();
    WorkList.pop_back();

    State OldIn = InMap[BB];
    State OldOut = OutMap[BB];

    State In = unknown;

    for (auto it = pred_begin(BB), et = pred_end(BB); it != et; ++it) {
      In = Join(In, OutMap[*it]);
    }

  if (debug){
    llvm::errs() << "\nInBLOCK " << BB->getName() << "\n";
    Dump(In); }

    State Out = ExecuteB(In, BB);

    if (debug){
    llvm::errs() << "OutBLOCK " << BB->getName() << "\n";
    Dump(Out);}

    if (Diff(In, OldIn) || Diff(Out, OldOut)) {
      if (debug){
      llvm::errs() << "ChangedBLOCK " << BB->getName() << "\n";}

      if (debug && Diff(Out, OldOut) && !Diff(In, OldIn)){
      llvm::errs() << "OUTPUT CHANGED BUT INPUT DIDN'T " << BB->getName() << "\n";}

      InMap[BB] = In;
      OutMap[BB] = Out;

      for (auto it = succ_begin(BB), et = succ_end(BB); it != et; ++it) {
        if (std::find(WorkList.begin(), WorkList.end(), *it) == WorkList.end()) {
          //llvm::errs() << "Adding " << (*it)->getName() << "\n";
          WorkList.push_back(*it);
        }
      }
    }
  }

  llvm::errs() << "ANALYSIS OF " << F.getName() << "\n";

  for (auto &BB: F) {
    State In = InMap[&BB];
    State Out = OutMap[&BB];
    llvm::errs() << "\nInBLOCK " << BB.getName() << "\n";
    Dump(In);

    BB.dump();

    llvm::errs() << "OutBLOCK " << BB.getName() << "\n";
    Dump(Out);
  }

  for (auto &BB: F) {
    State In = InMap[&BB];

    State SI = In;

    for (BasicBlock::iterator Iter = BB.begin(), E = BB.end(); Iter != E;) {
      Instruction *I = Iter++;

      if (auto C = dyn_cast<BinaryOperator>(I)) {
        if (C->getOperand(1) == Mask) {
          Instruction *PtrToInt = dyn_cast<Instruction>(C->getOperand(0));
          Value *BasePtr = PtrToInt->getOperand(0);

          if (GetRange(SI, BasePtr).allowed()) {
            Instruction *IntToPtr = Iter++; // Skip IntToPtr instruction;

            llvm::errs() << "\nRemoving CHECK ";
            C->dump();
            llvm::errs() << "\n";

            IntToPtr->replaceAllUsesWith(BasePtr);

            IntToPtr->eraseFromParent();
            C->eraseFromParent();
            PtrToInt->eraseFromParent();

            continue;
          }
        }
      }


      SI = ExecuteI(SI, I);
    }
  }

  llvm::errs() << "OptFunc " << F.getName() << "\n";
  F.dump();

  return true;
}

} // end anonymous namespace

char MemMask::ID = 0;
INITIALIZE_PASS_BEGIN(MemMask, "mem-mask",
                      "mem-mask instrumentation pass", false, false)
INITIALIZE_PASS_DEPENDENCY(AugmentArgs)
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
