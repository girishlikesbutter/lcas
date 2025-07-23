# Context Engineering Workflow

This directory contains all context engineering tasks for the LCAS project. Each task follows a structured workflow from initial requirements to implementation.

## Structure

```
context-engineering/
├── 01-<task-name>/                 # Tasks numbered by implementation order
│   ├── INITIAL_<task_name>.md      # Initial requirements
│   └── PRP_<task_name>.md          # Generated Product Requirements Prompt
├── templates/                       # Templates and examples
│   ├── INITIAL.md                  # Base INITIAL template
│   ├── INITIAL_EXAMPLE.md          # Example INITIAL file
│   ├── prp_base.md                 # Base PRP template
│   └── EXAMPLE_multi_agent_prp.md  # Example complete PRP
├── use-cases/                       # Advanced use case templates
│   ├── mcp-server/                 # MCP server template
│   ├── pydantic-ai/                # Pydantic AI agent template
│   └── template-generator/         # Meta-template generator
├── .claude/                         # Claude Code configuration
│   ├── commands/                   # Custom commands
│   └── settings.local.json         # Local settings
├── CLAUDE.md                        # Global AI assistant rules
├── CONTEXT_ENGINEERING_GUIDE.md     # Full methodology guide
├── IMPLEMENTATION_TIMELINE.md       # Chronological task progression
└── README.md                        # This file
```

## Current Tasks (Chronological Order)

Tasks are prefixed with numbers (01, 02, etc.) to show implementation order. See **IMPLEMENTATION_TIMELINE.md** for detailed progression.

### 01-surrogate-data-generator/
**Foundation** - Creates the base script structure and DataGenerator class for synthetic training data generation.

### 02-sample-uniform-sphere/
**Core Functionality** - Implements uniform sphere sampling for sun vector generation using Gaussian normalization.

### 03-ray-tracing-adaptation/
**Shadow Calculations** - Adds ray tracing capability to calculate lit fractions for each conceptual face.

### 04-data-generation-loop/ ✅ COMPLETED
**Data Collection** - Completes the pipeline with progress tracking (tqdm) and CSV output (numpy.savetxt).

## Workflow

1. **Create INITIAL file**: Define requirements in `INITIAL_<descriptive_name>.md`
   - Use `templates/INITIAL.md` as a starting point
   - See `templates/INITIAL_EXAMPLE.md` for reference

2. **Generate PRP**: Use `/generate-prp path/to/INITIAL_<name>.md` 
   - This creates a comprehensive Product Requirements Prompt
   - Uses `templates/prp_base.md` as the template

3. **Execute PRP**: Use `/execute-prp path/to/PRP_<name>.md`
   - Implements the feature based on the PRP

## Best Practices

- Keep INITIAL files detailed and specific
- Include examples, documentation links, and considerations
- Reference existing code patterns where applicable
- Update this README when adding new tasks
- Follow the rules in CLAUDE.md for consistent behavior

## Resources

- **CONTEXT_ENGINEERING_GUIDE.md** - Complete guide to the methodology
- **templates/** - Starting points for new tasks
- **use-cases/** - Advanced templates for specific frameworks