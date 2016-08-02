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
#include "llvm/Support/Timer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <sstream>

#include "Avery.h"

using namespace llvm;

#define DEBUG_TYPE "avery-mem-mask"

const int64_t GuardSize = 0x1000;

bool Avery::safePtr(Value *I, DenseSet<Value *> &prot, SmallSet<Value *, 8> &phis, int64_t offset, int64_t size, Value *&target, bool CanOffset) {
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

    // We can mask at the unwind location and the normal path however (how can we tell which is relevant here?)
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
Value *Avery::protectValue(Function &F, DenseSet<Value *> &prot, Use &PtrUse, Value *Mask, bool CanOffset) {
  IRBuilder<> IRB(&F.getEntryBlock());
  auto it = F.getEntryBlock().begin();
  ++it; // Skip Mask inline assembly
  IRB.SetInsertPoint(&F.getEntryBlock(), it);

  Value *Ptr = PtrUse.get();
  Value *Target;
  auto MemType = Ptr->getType()->getPointerElementType();
  SmallSet<Value *, 8> phis;

  // TODO: Have safePtr return the Target in a Use & which if updated ensures that PtrUse is safe
  // This avoids the issue with not being able to update constants uses later
  if (safePtr(Ptr, prot, phis, 0, CanOffset ? DL->getTypeAllocSize(MemType) : 0, Target, CanOffset))
    return nullptr;

  // TODO: If Target is a Constant, call safeConstantPtr which has a map of Constants to protected values
  // If a safe one is found, recreate the instructions required to get the constant,
  //    <- this requires runtime instructions, another AND might be just as fast
  //    <- bitcasts would be free
  // Add Target -> MaskedPtr to the constant map

  if (auto TI = dyn_cast<Instruction>(Target)) {
    // Insert it after the instruction generating the pointer
    BasicBlock::iterator it = std::next(BasicBlock::iterator(TI));
    
    // Skip prefix nodes
    while (isa<PHINode>(&*it) || isa<LandingPadInst>(&*it) || isa<CatchPadInst>(&*it)) ++it;

    IRB.SetInsertPoint(TI->getParent(), it);
  }

  auto PtrVal = IRB.Insert(CastInst::Create(Instruction::PtrToInt, Target, IntPtrTy), ""); // This must be an instruction
      //IRB.CreatePtrToInt(Target, IntPtrTy);
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

  return MaskedVal;
}

void Avery::protectValueAndSeg(Function &F, DenseSet<Value *> &prot, Instruction *I, unsigned PtrOp, Value *Mask) {
  protectValue(F, prot, I->getOperandUse(PtrOp), Mask, true);
  IRBuilder<> IRB(I);
  auto SegPtrVal = IRB.CreatePtrToInt(I->getOperand(PtrOp), IntPtrTy);
  auto SegPtr = IRB.CreateIntToPtr(SegPtrVal, I->getOperand(PtrOp)->getType()->getPointerElementType()->getPointerTo(256));
  //auto SegPtr = IRB.CreateAddrSpaceCast(I->getOperand(PtrOp), I->getOperand(PtrOp)->getType()->getPointerElementType()->getPointerTo(256)); //CRASHES
  I->setOperand(PtrOp, SegPtr);
}

void Avery::protectFunction(Function &F, DenseSet<Value *> &prot, Value *Mask) {
  for (auto &BB: F) {
    for (BasicBlock::iterator Iter = BB.begin(), E = BB.end(); Iter != E;) {
      Instruction *I = &*(Iter++);
      if (auto LI = dyn_cast<LoadInst>(I)) {
        protectValueAndSeg(F, prot, I, LI->getPointerOperandIndex(), Mask);
      } else if (auto SI = dyn_cast<StoreInst>(I)) {
        protectValueAndSeg(F, prot, I, SI->getPointerOperandIndex(), Mask);
      } else if (auto AI = dyn_cast<AtomicRMWInst>(I)) {
        protectValueAndSeg(F, prot, I, AI->getPointerOperandIndex(), Mask);
      } else if (auto AI = dyn_cast<AtomicCmpXchgInst>(I)) {
        protectValueAndSeg(F, prot, I, AI->getPointerOperandIndex(), Mask);
      } else if (auto RI = dyn_cast<ReturnInst>(I)) {
          auto Arg = RI->getReturnValue();
          if (Arg && Arg->getType()->isPointerTy())
            protectValue(F, prot, RI->getOperandUse(0), Mask, false);
      } else if (dyn_cast<FenceInst>(I)) {
      } else if (dyn_cast<InvokeInst>(I)) {
      } else if (auto MI = dyn_cast<MemIntrinsic>(I)) {
        protectValueAndSeg(F, prot, MI, 0, Mask);
        Type *Tys[] = { MI->getRawDest()->getType(), nullptr, nullptr };
        ArrayRef<Type *> Args;
        if (auto MTI = dyn_cast<MemTransferInst>(MI)) {
          protectValueAndSeg(F, prot, MI, 1, Mask);
          Tys[1] = MTI->getRawSource()->getType();
          Tys[2] = MI->getLength()->getType();
          Args = ArrayRef<Type *>(Tys, 3);
        } else {
          Tys[1] = MI->getLength()->getType();
          Args = ArrayRef<Type *>(Tys, 2);
        }
        auto NewFn = Intrinsic::getDeclaration(F.getParent(), MI->getIntrinsicID(), Args);
        MI->mutateFunctionType(dyn_cast<FunctionType>(NewFn->getType()->getElementType()));
        MI->setCalledFunction(NewFn);
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

void Avery::memMask(Function &F) {
  if (F.empty()) {
    return;
  }
  IRBuilder<> IRB(&F.front(), F.begin()->getFirstInsertionPt());
  auto Asm = InlineAsm::get(FunctionType::get(IntPtrTy, false), "", "={r15}", false, false, InlineAsm::AD_Intel);
  Value *Mask = IRB.CreateCall(Asm);

  DenseSet<Value *> prot;

  {
    NamedRegionTimer T("Protection", "Avery", TimePassesIsEnabled);
    protectFunction(F, prot, Mask);
  }
}

