//===-- CFGMST.h - Minimum Spanning Tree for CFG ----------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a Union-find algorithm to compute Minimum Spanning Tree
// for a given CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <string>
#include <utility>
#include <vector>

namespace llvm {

class Avery : public ModulePass {
 public:
  explicit Avery()
      : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
  static char ID;
  const char *getPassName() const override { return "Avery"; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
  }

 private:
  struct Range;

  Function *Func;
  Type *IntPtrTy;
  const DataLayout *DL;

  void augmentArgs(Module &M);
  void memMask(Function &F);
  bool eliminateMasks(Function &F, DenseSet<Value *> &prot);

  typedef DenseMap<Value *, Range> State;

  void ExecuteI(State &R, Instruction *I, bool Widen);
  void ExecuteB(State &InState, BasicBlock *BB, bool Widen);
  void Join(State &A, State &B);

  void protectFunction(Function &F, DenseSet<Value *> &prot, Value *Mask);
  bool safePtr(Value *I, DenseSet<Value *> &prot, SmallSet<Value *, 8> &phis, int64_t offset, int64_t size, Value *&target, bool CanOffset);
  void protectValueAndSeg(Function &F, DenseSet<Value *> &prot, Instruction *I, unsigned PtrOp, Value *Mask);
  Value *protectValue(Function &F, DenseSet<Value *> &prot, Use &PtrUse, Value *Mask, bool CanOffset);
};

} // end namespace llvm
