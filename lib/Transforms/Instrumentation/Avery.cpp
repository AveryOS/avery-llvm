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

#define DEBUG_TYPE "avery"

bool Avery::runOnModule(Module &M) {
  if (Triple(M.getTargetTriple()).getOS() != Triple::OSType::Avery)
    return false;

  DL = &M.getDataLayout();
  StackPtrTy = Type::getInt8PtrTy(M.getContext());
  IntPtrTy = DL->getIntPtrType(M.getContext());
  Int32Ty = Type::getInt32Ty(M.getContext());

  augmentArgs(M);

  for (auto Iter = M.begin(), E = M.end(); Iter != E; ) {
    Function &F = *(Iter++);
    memMask(F);
    splitStacks(F);
  }

  return true;
}

char Avery::ID = 0;
INITIALIZE_PASS(
    Avery, "avery",
    "AveryPass"
    "ModulePass",
    false, false)
ModulePass *llvm::createAveryPass() {
  return new Avery();
}

