//===-- AverySafeStack.cpp - Safe Stack Insertion ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass splits the stack into the safe stack (kept as-is for LLVM backend)
// and the unsafe stack (explicitly allocated and managed through the runtime
// support library).
//
// http://clang.llvm.org/docs/SafeStack.html
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include "Avery.h"

using namespace llvm;

#define DEBUG_TYPE "datastack"

/// Unsafe stack alignment. Each stack frame must ensure that the stack is
/// aligned to this value. We need to re-align the unsafe stack if the
/// alignment of any object on the stack exceeds this value.
///
/// 16 seems like a reasonable upper bound on the alignment of objects that we
/// might expect to appear on the stack on most common targets.
enum { StackAlignment = 16 };

uint64_t Avery::getStaticAllocaAllocationSize(const AllocaInst* AI) {
  uint64_t Size = DL->getTypeAllocSize(AI->getAllocatedType());
  if (AI->isArrayAllocation()) {
    auto C = dyn_cast<ConstantInt>(AI->getArraySize());
    if (!C)
      return 0;
    Size *= C->getZExtValue();
  }
  return Size;
}

void Avery::findInsts(Function &F,
                          SmallVectorImpl<AllocaInst *> &StaticAllocas,
                          SmallVectorImpl<AllocaInst *> &DynamicAllocas,
                          SmallVectorImpl<Argument *> &ByValArguments) {
  for (Instruction &I : instructions(&F)) {
    if (auto AI = dyn_cast<AllocaInst>(&I)) {
      if (AI->isStaticAlloca()) {
        StaticAllocas.push_back(AI);
      } else {
        DynamicAllocas.push_back(AI);
      }
    } else if (auto II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::gcroot)
        llvm::report_fatal_error(
            "gcroot intrinsic not compatible with safestack attribute");
    }
  }
  for (Argument &Arg : F.args()) {
    if (!Arg.hasByValAttr())
      continue;
    ByValArguments.push_back(&Arg);
  }
}

AllocaInst *
Avery::createStackRestorePoints(Function &F,
                                    Value *StaticTop, bool NeedDynamicTop) {
  // We need the current value of the shadow stack pointer to restore
  // after longjmp or exception catching.

  IRBuilder<> IRB(&F.front(), F.begin()->getFirstInsertionPt());

  AllocaInst *DynamicTop = nullptr;
  if (NeedDynamicTop)
    // If we also have dynamic alloca's, the stack pointer value changes
    // throughout the function. For now we store it in an alloca.
    DynamicTop = IRB.CreateAlloca(StackPtrTy, /*ArraySize=*/nullptr,
                                  "unsafe_stack_dynamic_ptr");

  if (NeedDynamicTop)
    IRB.CreateStore(StaticTop, DynamicTop);

  return DynamicTop;
}

Value *Avery::moveStaticAllocasToUnsafeStack(
    Function &F, ArrayRef<AllocaInst *> StaticAllocas,
    ArrayRef<Argument *> ByValArguments) {
  auto args = F.arg_begin();
  Value *BasePointer = &*args;
  assert(BasePointer->getType() == StackPtrTy);

  if (StaticAllocas.empty() && ByValArguments.empty())
    return BasePointer;


  DIBuilder DIB(*F.getParent());

  // We explicitly compute and set the unsafe stack layout for all unsafe
  // static alloca instructions. We save the unsafe "base pointer" in the
  // prologue into a local variable and restore it in the epilogue.

  // Compute maximum alignment among static objects on the unsafe stack.
  unsigned MaxAlignment = 0;
  for (Argument *Arg : ByValArguments) {
    Type *Ty = Arg->getType()->getPointerElementType();
    unsigned Align = std::max((unsigned)DL->getPrefTypeAlignment(Ty),
                              Arg->getParamAlignment());
    if (Align > MaxAlignment)
      MaxAlignment = Align;
  }
  for (AllocaInst *AI : StaticAllocas) {
    Type *Ty = AI->getAllocatedType();
    unsigned Align =
        std::max((unsigned)DL->getPrefTypeAlignment(Ty), AI->getAlignment());
    if (Align > MaxAlignment)
      MaxAlignment = Align;
  }

  if (MaxAlignment > StackAlignment) {
    report_fatal_error("Too large alignment on data stack");
      /*
    // Re-align the base pointer according to the max requested alignment.
    assert(isPowerOf2_32(MaxAlignment));
    IRB.SetInsertPoint(BasePointer->getNextNode());
    BasePointer = cast<Instruction>(IRB.CreateIntToPtr(
        IRB.CreateAnd(IRB.CreatePtrToInt(BasePointer, IntPtrTy),
                      ConstantInt::get(IntPtrTy, ~uint64_t(MaxAlignment - 1))),
        StackPtrTy));*/
  }

  int64_t StaticOffset = 0; // Current stack top.

  for (Argument *Arg : ByValArguments) {
    IRBuilder<> IRB(&F.front(), F.begin()->getFirstInsertionPt());

    Type *Ty = Arg->getType()->getPointerElementType();

    uint64_t Size = DL->getTypeStoreSize(Ty);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    // Ensure the object is properly aligned.
    unsigned Align = std::max((unsigned)DL->getPrefTypeAlignment(Ty),
                              Arg->getParamAlignment());

    Value *Off = IRB.CreateGEP(BasePointer, // BasePointer is i8*
                               ConstantInt::get(Int32Ty, StaticOffset));
    Value *NewArg = IRB.CreateBitCast(Off, Arg->getType(),
                                     Arg->getName() + ".unsafe-byval");

    // Replace alloc with the new location.
    replaceDbgDeclare(Arg, BasePointer, &*IRB.GetInsertPoint(), DIB,
                      /*Deref=*/true, StaticOffset);
    Arg->replaceAllUsesWith(NewArg);
    IRB.SetInsertPoint(cast<Instruction>(NewArg)->getNextNode());
    IRB.CreateMemCpy(Off, Arg, Size, Arg->getParamAlignment());

    // Add alignment.
    // NOTE: we ensure that BasePointer itself is aligned to >= Align.
    StaticOffset += Size;
    StaticOffset = RoundUpToAlignment(StaticOffset, Align);
  }

  // Allocate space for every unsafe static AllocaInst on the unsafe stack.
  for (AllocaInst *AI : StaticAllocas) {
    IRBuilder<> IRB(AI);

    Type *Ty = AI->getAllocatedType();
    uint64_t Size = getStaticAllocaAllocationSize(AI);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    // Ensure the object is properly aligned.
    unsigned Align =
        std::max((unsigned)DL->getPrefTypeAlignment(Ty), AI->getAlignment());

    Value *Off = IRB.CreateGEP(BasePointer, // BasePointer is i8*
                               ConstantInt::get(Int32Ty, StaticOffset));
    Value *NewAI = IRB.CreateBitCast(Off, AI->getType(), AI->getName());
    if (AI->hasName() && isa<Instruction>(NewAI))
      cast<Instruction>(NewAI)->takeName(AI);

    // Replace alloc with the new location.
    replaceDbgDeclareForAlloca(AI, BasePointer, DIB, /*Deref=*/true, StaticOffset);
    AI->replaceAllUsesWith(NewAI);
    AI->eraseFromParent();

    // Add alignment.
    // NOTE: we ensure that BasePointer itself is aligned to >= Align.
    StaticOffset += Size;
    StaticOffset = RoundUpToAlignment(StaticOffset, Align);
  }

  // Re-align BasePointer so that our callees would see it aligned as
  // expected.
  // FIXME: no need to update BasePointer in leaf functions.
  StaticOffset = RoundUpToAlignment(StaticOffset, StackAlignment);

  F.addFnAttr("data-stack-size", std::to_string(StaticOffset));

  return BasePointer;
}

void Avery::moveDynamicAllocasToUnsafeStack(
    Function &F, AllocaInst *DynamicTop,
    ArrayRef<AllocaInst *> DynamicAllocas) {
  DIBuilder DIB(*F.getParent());

  if (DynamicAllocas.empty()) {
    return;
  }

  report_fatal_error("Dynamic allocas not supported");
  /*
  for (AllocaInst *AI : DynamicAllocas) {
    IRBuilder<> IRB(AI);

    // Compute the new SP value (after AI).
    Value *ArraySize = AI->getArraySize();
    if (ArraySize->getType() != IntPtrTy)
      ArraySize = IRB.CreateIntCast(ArraySize, IntPtrTy, false);

    Type *Ty = AI->getAllocatedType();
    uint64_t TySize = DL->getTypeAllocSize(Ty);
    Value *Size = IRB.CreateMul(ArraySize, ConstantInt::get(IntPtrTy, TySize));

    Value *SP = IRB.CreatePtrToInt(IRB.CreateLoad(UnsafeStackPtr), IntPtrTy);
    SP = IRB.CreateSub(SP, Size);

    // Align the SP value to satisfy the AllocaInst, type and stack alignments.
    unsigned Align = std::max(
        std::max((unsigned)DL->getPrefTypeAlignment(Ty), AI->getAlignment()),
        (unsigned)StackAlignment);

    assert(isPowerOf2_32(Align));
    Value *NewTop = IRB.CreateIntToPtr(
        IRB.CreateAnd(SP, ConstantInt::get(IntPtrTy, ~uint64_t(Align - 1))),
        StackPtrTy);

    // Save the stack pointer.
    IRB.CreateStore(NewTop, UnsafeStackPtr);
    if (DynamicTop)
      IRB.CreateStore(NewTop, DynamicTop);

    Value *NewAI = IRB.CreatePointerCast(NewTop, AI->getType());
    if (AI->hasName() && isa<Instruction>(NewAI))
      NewAI->takeName(AI);

    replaceDbgDeclareForAlloca(AI, NewAI, DIB, /*Deref=*//*true);
    AI->replaceAllUsesWith(NewAI);
    AI->eraseFromParent();
  }

  // Now go through the instructions again, replacing stacksave/stackrestore.
  for (inst_iterator It = inst_begin(&F), Ie = inst_end(&F); It != Ie;) {
    Instruction *I = &*(It++);
    auto II = dyn_cast<IntrinsicInst>(I);
    if (!II)
      continue;

    if (II->getIntrinsicID() == Intrinsic::stacksave) {
      IRBuilder<> IRB(II);
      Instruction *LI = IRB.CreateLoad(UnsafeStackPtr);
      LI->takeName(II);
      II->replaceAllUsesWith(LI);
      II->eraseFromParent();
    } else if (II->getIntrinsicID() == Intrinsic::stackrestore) {
      IRBuilder<> IRB(II);
      Instruction *SI = IRB.CreateStore(II->getArgOperand(0), UnsafeStackPtr);
      SI->takeName(II);
      assert(II->use_empty());
      II->eraseFromParent();
    }
  }*/
}

void Avery::splitStacks(Function &F) {
  DEBUG(dbgs() << "[SafeStack] Function: " << F.getName() << "\n");

  if (F.isDeclaration()) {
    return;
  }

  SmallVector<AllocaInst *, 16> StaticAllocas;
  SmallVector<AllocaInst *, 4> DynamicAllocas;
  SmallVector<Argument *, 4> ByValArguments;

  // Find all static and dynamic alloca instructions that must be moved to the
  // unsafe stack, all return instructions and stack restore points.
  findInsts(F, StaticAllocas, DynamicAllocas, ByValArguments);

  if (StaticAllocas.empty() && DynamicAllocas.empty() &&
      ByValArguments.empty())
    return; // Nothing to do in this function.

  // The top of the unsafe stack after all unsafe static allocas are allocated.
  Value *StaticTop = moveStaticAllocasToUnsafeStack(F, StaticAllocas,
                                                    ByValArguments);

  // Safe stack object that stores the current unsafe stack top. It is updated
  // as unsafe dynamic (non-constant-sized) allocas are allocated and freed.
  // This is only needed if we need to restore stack pointer after longjmp
  // or exceptions, and we have dynamic allocations.
  // FIXME: a better alternative might be to store the unsafe stack pointer
  // before setjmp / invoke instructions.
  AllocaInst *DynamicTop = createStackRestorePoints(
      F, StaticTop, !DynamicAllocas.empty());

  // Handle dynamic allocas.
  moveDynamicAllocasToUnsafeStack(F, DynamicTop,
                                  DynamicAllocas);

  DEBUG(dbgs() << "[SafeStack]     safestack applied\n");
}

