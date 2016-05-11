#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/APSInt.h"
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

#define DEBUG_TYPE "memmask"

struct Range {
  bool Unknown;
  int64_t RelStart;
  int64_t RelEnd;

  static Range top() {
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

  Range offset(int64_t o) const {
    Range r = *this;

    if (o == 0)
      return r;

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

Range GetRange(const MemMask::State &S, Value *V) {
   auto it = S.find(V);
   if (it == S.end()) {
     return Range::top();
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

void MemMask::ExecuteI(State &R, Instruction *I, bool Widen) {
  if (auto C = dyn_cast<CastInst>(I)) {
    if (C->isNoopCast(*DL)) {
      R[C] = GetRange(R, C->getOperand(0));
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
    R[Val] = r;
    R[I] = Range::top();
  } else if (auto PN = dyn_cast<PHINode>(I)) {
    Range phi = R[PN];
    for (Value *val: PN->incoming_values()) {
      phi = JoinRange(phi, GetRange(R, val));
    }
    if (Widen && (R[PN] != phi)) {
      phi = Range::top();
    }
    R[PN] = phi;
  } else if (R.find(I) != R.end()) {
    R[I] = Range::top();
  }
}

void MemMask::ExecuteB(State &State, BasicBlock *BB, bool Widen) {
  for (auto &I: *BB) {
    ExecuteI(State, &I, Widen);
  }
}

void MemMask::Join(MemMask::State &A, MemMask::State &B) {
  for (auto &v: A) {
    v.getSecond() = JoinRange(v.getSecond(), B[v.getFirst()]);
  }
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

struct BlockState {
  unsigned Visits;
  Avery::State In;
  Avery::State Out;
};

bool Avery::eliminateMasks(Function &F, DenseSet<Value *> &prot) {
  Mask = &*F.arg_begin();

  bool debug = false;

  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();

  DenseMap<Value *, Value *> checks;

  return true;

  if (LI->empty())
    return true;

  {NamedRegionTimer T("Loop check duplication", "MemMask", TimePassesIsEnabled);

  for (auto &BB: F) {
    Loop *L = LI->getLoopFor(&BB);
    if(!L)
      continue;

    for (BasicBlock::iterator Iter = BB.begin(), E = BB.end(); Iter != E;) {
      Instruction *I = &*(Iter++);

      if (auto C = dyn_cast<BinaryOperator>(I)) {
        if (C->getOperand(1) == Mask) {
          Instruction *PtrToInt = dyn_cast<Instruction>(C->getOperand(0));
          Value *BasePtr = PtrToInt->getOperand(0);

          auto PN = dyn_cast<PHINode>(BasePtr);
          if (!PN)
            continue;

          auto protectIncoming = [&] () -> Value * {
            for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i) {
              if (LI->getLoopFor(PN->getIncomingBlock(i)) != L) {

                auto NC = protectValue(F, prot, PN->getOperandUse(i), Mask, true);
                if (NC) {
                  if (debug) {
                    llvm::errs() << "\nINSERTED CHECK in LOOP for ";
                    PN->getOperand(i)->dump();
                    llvm::errs() << "\nBASED ON CHECK";
                    C->dump();
                    llvm::errs() << "\n";
                  }
                  return NC;
                }
              }
            }
            return nullptr;
          };

          auto NC = protectIncoming();

          if (NC) {
            checks.insert(std::pair<Value *, Value *>(C, NC));
          }
        }
      }
    }
  }

  }

  DenseMap<BasicBlock *, BlockState> BlockMap;

  {NamedRegionTimer T("Analysis", "MemMask", TimePassesIsEnabled);

  State unknown;

  for (auto &BB: F) {
    for (auto &I: BB) {
      if (!I.getType()->isVoidTy())
        unknown.insert(std::pair<Value *, Range>(&I, Range::unknown()));
    }
  }

  std::vector<BasicBlock *> WorkList;


  for (auto &BB: F) {
    WorkList.insert(WorkList.begin(), &BB);
    BlockState s;
    s.In = unknown;
    s.Out = unknown;
    s.Visits = 0;
    BlockMap.insert(std::pair<BasicBlock *, BlockState>(&BB, s));
  }

  if (debug) {
    llvm::errs() << "DURING ANALYSIS OF " << F.getName() << "\n";
    F.dump();
  }

  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.back();
    WorkList.pop_back();

    auto &BS = BlockMap[BB];

    State In = unknown;

    for (auto it = pred_begin(BB), et = pred_end(BB); it != et; ++it) {
      Join(In, BlockMap[*it].Out);
    }

  if (debug){
    llvm::errs() << "\nInBLOCK " << BB->getName() << "\n";
    Dump(In); }

/*
    if (BS.Visits != 0 && Diff(Out, BS.In) && !Diff(In, BS.In)) {
      llvm::errs() << "OUTPUT CHANGED BUT INPUT DIDN'T " << BB->getName() << "\n";
    }
*/
    if (Diff(In, BS.In) || BS.Visits == 0) {
      if (debug){
      llvm::errs() << "ChangedBLOCK " << BB->getName() << "\n";}

      BS.In = In;
      BS.Out = std::move(In);

      ExecuteB(BS.Out, BB, BS.Visits > 2);

      if (debug){
      llvm::errs() << "OutBLOCK " << BB->getName() << "\n";
      Dump(BS.Out);}


      for (auto it = succ_begin(BB), et = succ_end(BB); it != et; ++it) {
        if (std::find(WorkList.begin(), WorkList.end(), *it) == WorkList.end()) {
          //llvm::errs() << "Adding " << (*it)->getName() << "\n";
          WorkList.push_back(*it);
        }
      }
    }

    BS.Visits++;
  }
if (debug) {
  llvm::errs() << "ANALYSIS OF " << F.getName() << "\n";

  for (auto &BB: F) {
    auto &BS = BlockMap[&BB];
    llvm::errs() << "\nInBLOCK " << BB.getName() << "\n";
    Dump(BS.In);

    BB.dump();

    llvm::errs() << "OutBLOCK " << BB.getName() << "\n";
    Dump(BS.Out);
  }
}


  }

  {NamedRegionTimer T("Check removing", "MemMask", TimePassesIsEnabled);

  std::vector<Value *> removing;

  for (auto &BB: F) {
    State &SI = BlockMap[&BB].In;

    for (BasicBlock::iterator Iter = BB.begin(), E = BB.end(); Iter != E;) {
      Instruction *I = &*(Iter++);
/*
      llvm::errs() << "\nEXEC INSTR in " << BB.getName() << "\n";
      I->dump();
      llvm::errs() << "--\n";
      Dump(SI);
*/
      if (auto C = dyn_cast<BinaryOperator>(I)) {
        if (C->getOperand(1) == Mask) {
          Instruction *PtrToInt = dyn_cast<Instruction>(C->getOperand(0));
          assert(PtrToInt != nullptr);
          Value *BasePtr = PtrToInt->getOperand(0);

          if (GetRange(SI, BasePtr).allowed()) {
            llvm::errs() << "\nRemoving CHECK in " << F.getName();
            C->dump();
            llvm::errs() << "\n";
/*
            llvm::errs() << "\nProtecting ";
            BasePtr->dump();
            llvm::errs() << "\n";

            llvm::errs() << "\nWith range";
            llvm::errs() << GetRange(SI, BasePtr).str() << "\n";
*/
            removing.push_back(C);
            checks.erase(C);
          }
        }
      }


      ExecuteI(SI, I, false);
    }
  }

  for (auto &pair: checks) {
    auto C = dyn_cast<Instruction>(pair.getSecond());
    removing.push_back(C);

    if (debug) {
    llvm::errs() << "\nRemoving UNHELPFUL CHECK in " << F.getName();
    C->dump();
    llvm::errs() << "\n";}
  }

  for (auto &r: removing) {
    auto C = dyn_cast<Instruction>(r);

    Instruction *PtrToInt = dyn_cast<Instruction>(C->getOperand(0));
    Instruction *IntToPtr = dyn_cast<Instruction>(C->use_begin()->getUser());
    Value *BasePtr = PtrToInt->getOperand(0);
/*
    llvm::errs() << "\nPtrToInt CHECK in " << F.getName();
    PtrToInt->dump();
    llvm::errs() << "\n";

    llvm::errs() << "\nAnd CHECK in " << F.getName();
    C->dump();
    llvm::errs() << "\n";

    llvm::errs() << "\nIntToPtr CHECK in " << F.getName();
    IntToPtr->dump();
    llvm::errs() << "\n";
*/
    IntToPtr->replaceAllUsesWith(BasePtr);

    IntToPtr->eraseFromParent();
    C->eraseFromParent();
    PtrToInt->eraseFromParent();
  }

  }

  if (debug) {
  llvm::errs() << "OptFunc " << F.getName() << "\n";
  F.dump();
}
  return true;
}
