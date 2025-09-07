# llmentary - REVISED Development Workflow

**Project:** LLM Regression Testing Tool
**New Approach:** Training/Testing Workflow (not passive monitoring)
**Goal:** Create a deliberate, user-controlled regression testing system for LLM applications

## 🎯 Revised Project Vision

Transform llmentary from a passive monitoring tool into an **active regression testing framework** where developers:

1. **Train** by running their app and saving Q&A pairs they approve of
2. **Test** by running regression tests against saved baselines
3. **Integrate** with minimal code changes (decorators/context managers)

---

## 📋 New Implementation Plan

### Phase 1: Core Training/Testing Framework ✅ **COMPLETED**
**Priority:** Critical - Complete pivot of approach
**Status:** ✅ 5/5 completed

**Deliverables:**
- [x] Design new database schema (questions plaintext, answers hashed)
- [x] Implement Training Mode with interactive approval workflow
- [x] Create Test Mode for regression testing
- [x] Build minimal integration patterns (decorators, context managers)
- [x] Update CLI for train/test commands

**Acceptance Criteria:**
- ✅ User can run app in training mode and selectively save Q&A pairs
- ✅ Test mode compares current responses to approved baselines
- ✅ <3 lines of code to integrate into existing apps
- ✅ Questions stored in plaintext, answers securely hashed

---

### Phase 2: Enhanced Testing Features ⏸️ **PENDING**
**Priority:** High - Make testing robust
**Status:** 🚫 0/4 completed

**Deliverables:**
- [ ] Add semantic similarity for flexible answer matching
- [ ] Implement test result reporting and diff visualization  
- [ ] Create test filtering (by tags, date, model, etc.)
- [ ] Add batch testing capabilities

---

### Phase 3: Production Integration ⏸️ **PENDING**
**Priority:** Medium - Ease of use
**Status:** 🚫 0/3 completed

**Deliverables:**
- [ ] Framework-specific integrations (FastAPI, Flask, etc.)
- [ ] CI/CD integration examples
- [ ] Configuration management and test organization

---

## 🔄 Key Design Changes

### **From: Passive Monitoring**
```python
# Old approach - logs everything automatically
AutoInstrument.auto_patch_all()
monitor.configure(store_raw_text=False)
# All LLM calls automatically monitored
```

### **To: Training/Testing Workflow**
```python
# New approach - deliberate training and testing
with llmentary.training_mode():
    response = llm.ask("What is our refund policy?")
    # User prompted: Save this Q&A? (y/n)
    
# Later: regression testing
llmentary.test_all()  # Validates all saved Q&A pairs
```

### **Security Model**
- **Questions**: Stored in plaintext (needed for test execution)
- **Answers**: Hashed with salt (privacy-preserving comparison)  
- **Local**: SQLite database in project directory
- **User Control**: Nothing saved without explicit approval

### **Integration Patterns**
```python
# Option 1: Decorator
@llmentary.trainable
def ask_support_question(question):
    return llm.ask(question)

# Option 2: Context Manager  
with llmentary.training_session("support"):
    answer = llm.ask(question)
    
# Option 3: Manual
llmentary.capture_interaction(question, answer, category="support")
```

---

## 📊 Progress Tracking

**Overall Progress:** Phase 1 Complete (100% of core framework)

**Current Status:** ✅ MAJOR PIVOT SUCCESSFUL
**Achievement:** Complete training/testing framework with CLI and integrations
**Next Phase:** Enhanced testing features (semantic similarity, advanced reporting)

---

## 🧪 Testing Strategy

**Phase 1 Testing:**
- Manual testing of training workflow
- Regression test validation accuracy
- Integration ease with sample apps

**Success Metrics:**
- <30 seconds to add to existing app
- >95% accuracy in detecting changed responses  
- Clear, actionable test failure reports

---

*This represents a major pivot from passive monitoring to active regression testing*
*Focus on user control, security, and minimal integration overhead*