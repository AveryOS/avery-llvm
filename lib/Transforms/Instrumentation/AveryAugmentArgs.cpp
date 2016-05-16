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

#define DEBUG_TYPE "avery-augment-args"

Function *RecreateFunction(Function *Func, FunctionType *NewType) {
  Function *NewFunc = Function::Create(NewType, Func->getLinkage());

  Func->getParent()->getFunctionList().insert(Module::FunctionListType::iterator(Func), NewFunc);
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
  unsigned NumArgs = Func->arg_size();
  for (unsigned i = 1, j = 3; i < NumArgs+1; i++, j++) {
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

void Avery::augmentArgs(Module &M) {
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
    ArgTypes.push_back(StackPtrTy);
    for (auto Type : F.getFunctionType()->params()) {
      ArgTypes.push_back(Type);
    }
    auto NFTy = FunctionType::get(F.getReturnType(), ArgTypes, F.isVarArg());

    auto &NF = *RecreateFunction(&F, NFTy);

    AttrBuilder Attrs;

    Attrs.addAttribute("no-frame-pointer-elim");
    Attrs.addAttribute("no-frame-pointer-elim-non-leaf");

    auto AS = NF.getAttributes().removeAttributes(M.getContext(), AttributeSet::FunctionIndex, Attrs);
    AS = AS.removeAttribute(F.getContext(), 3, Attribute::StructRet);
    AS = AS.removeAttribute(F.getContext(), 4, Attribute::StructRet);
    NF.setAttributes(AS);

    auto NewArg = NF.arg_begin();
    NewArg->setName("Mask");
    Value *Mask = &*NewArg;
    ++NewArg;
    NewArg->setName("Stack");
    Value *Stack = &*NewArg;
    ++NewArg;
    for (auto &Arg : F.args()) {
      Arg.replaceAllUsesWith(&*NewArg);
      NewArg->takeName(&Arg);
      ++NewArg;
    }

    F.eraseFromParent();

    NF.setCallingConv(CallingConv::Sandbox);

    for (auto &BB: NF) {
      for (BasicBlock::iterator Iter = BB.begin(), E = BB.end(); Iter != E;) {
        Instruction &I = *(Iter++);

        if (auto OldCall = dyn_cast<InvokeInst>(&I)) {
          FunctionType *CallFTy = cast<FunctionType>(
              OldCall->getCalledValue()->getType()->getPointerElementType());

          SmallVector<Type *, 8> ArgTypes;
          ArgTypes.push_back(IntPtrTy);
          ArgTypes.push_back(StackPtrTy);
          for (auto Type : CallFTy->params()) {
            ArgTypes.push_back(Type);
          }
          auto NCallFTy = FunctionType::get(CallFTy->getReturnType(), ArgTypes, CallFTy->isVarArg());

          SmallVector<Value *, 8> Args;
          Args.push_back(Mask);
          Args.push_back(Stack);
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
          if (isa<InlineAsm>(OldCall->getCalledValue()))
            continue;

          FunctionType *CallFTy = cast<FunctionType>(
              OldCall->getCalledValue()->getType()->getPointerElementType());

          SmallVector<Type *, 8> ArgTypes;
          ArgTypes.push_back(IntPtrTy);
          ArgTypes.push_back(StackPtrTy);
          for (auto Type : CallFTy->params()) {
            ArgTypes.push_back(Type);
          }
          auto NCallFTy = FunctionType::get(CallFTy->getReturnType(), ArgTypes, CallFTy->isVarArg());

          SmallVector<Value *, 8> Args;
          Args.push_back(Mask);
          Args.push_back(Stack);
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
  }
/*

    llvm::errs() << "ADDCALL" << NF.getName() << "\n";
    M.dump();
    llvm::errs() << "/ADDCALL\n";
*/
/*
    llvm::errs() << "PROTF" << NF.getName() << "\n";
    M.dump();
    llvm::errs() << "/PROTF\n";*/

/*
    llvm::errs() << "DONEDUMP\n";
    M.dump();
    llvm::errs() << "/DONEDUMP\n";*/
}
