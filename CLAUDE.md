# llmentary - Claude AI Assistant Instructions

## 🚀 ACTIVE WORKFLOW
**CRITICAL:** Follow the structured plan in `/workflow.md` for this project:
- **Project:** Privacy-First LLM Monitoring and Drift Detection Tool
- **Current Stage:** Stage 4 - Multiple Storage Backends
- **Status:** READY TO START (0/4 deliverables completed)
- **Next Actions:** Abstract storage interface, implement Redis/PostgreSQL backends, add connection pooling

**WORKFLOW REQUIREMENTS:**
1. MUST follow the workflow stages in sequential order
2. MUST update stage status in workflow.md when completing each deliverable
3. MUST check off deliverables as they are completed
4. MUST maintain stage tracking for easy resumption
5. MUST ensure all acceptance criteria are met before moving to next stage

When resuming work, ALWAYS check workflow.md first to see current progress.

## Project Context
This is llmentary - a privacy-first LLM monitoring and drift detection tool. The project has a solid foundation but needs significant feature implementation to match the documented capabilities.

**Key Principles:**
- Privacy-first design (SHA-256 hashing by default)
- Zero-effort integration (3 lines of code)
- Production-ready performance
- Comprehensive auto-instrumentation

## Current Implementation Status
- ✅ Basic monitoring and drift detection
- ✅ SQLite storage backend
- ❌ Modern LLM provider integration (critical issue)
- ❌ CLI interface (major missing feature)
- ❌ Advanced drift detection with semantic analysis
- ❌ Multiple storage backends
- ❌ Alerting system
- ❌ Performance optimizations

## Development Guidelines
1. **Follow workflow.md stages strictly** - Don't skip ahead
2. **Update progress immediately** - Mark deliverables complete as you finish them
3. **Test thoroughly** - Each stage has acceptance criteria that must be met
4. **Maintain backward compatibility** where possible
5. **Document as you go** - Update README.md for new features

## Stage 4 Focus Areas
Currently working on Multiple Storage Backends:
- Abstract storage interface for pluggable backends
- Implement Redis storage backend
- Implement PostgreSQL storage backend
- Add connection pooling and batch operations

**Stage 1 COMPLETED:** ✅ All major LLM providers now properly instrumented with modern APIs
**Stage 2 COMPLETED:** ✅ Full CLI interface with report, consistency, inspect, config, and status commands  
**Stage 3 COMPLETED:** ✅ Advanced drift detection with semantic similarity, configurable thresholds, and severity classification

---
*Last Updated: 2025-09-06*
*Follow workflow.md for all development decisions*