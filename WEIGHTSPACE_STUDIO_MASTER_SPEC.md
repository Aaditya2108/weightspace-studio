# WEIGHTSPACE STUDIO
## Master Project Specification & Development Handoff

**Project:** WeightSpace Studio  
**Version:** 0.1  
**Status:** Initial Development  
**Project Type:** Open-source developer framework + visual IDE

---

# 1. Project Vision

Build a general-purpose software framework that enables programmers to directly inspect, construct, manipulate, analyze and execute neural network weights.

The framework will introduce Weight-Space Programming (WSP) as a programming paradigm.

Instead of relying exclusively on conventional gradient-based training, developers should be able to write explicit programs that perform mathematical transformations on neural network parameters.

The framework must support both manual programming and AI-assisted weight-space programming.

# 2. Product Components

The product consists of:

1. Python SDK — programming interface.
2. WSP DSL — custom weight-space programming language.
3. WeightSpace Compiler — translates programs into executable tensor transformations.
4. Weight Engine — performs actual weight modifications.
5. Analysis Engine — analyzes tensors, layers and activations.
6. Runtime Engine — executes neural networks.
7. Evaluation Engine — measures the effects of modifications.
8. Visual IDE — graphical development environment.
9. AI Agent — assists programmers in creating and evaluating weight-space programs.

# 3. Product Scope

The framework must be designed for general neural networks.

Target architectures:

- Multilayer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers
- Large Language Models
- Diffusion models
- Reinforcement-learning policies

Architecture-specific functionality must be implemented through adapters.

The core framework must remain architecture-independent.

# 4. Technology Stack

## Core
- Python
- PyTorch
- NumPy
- Safetensors
- Pydantic

## Compiler
- Lark
- Python AST where appropriate
- Custom intermediate representation (IR)

## Frontend
- React
- TypeScript
- Vite
- Plotly
- D3.js

## Backend
- FastAPI
- SQLite

## Testing
- Pytest

## Infrastructure
- Git
- GitHub
- Docker

# 5. Core Architecture

The framework must contain these modules:

packages/weightspace/

- core/
- io/
- backends/
- operations/
- programming/
- compiler/
- engine/
- analysis/
- runtime/
- evaluation/
- adapters/
- agents/
- cli/

apps/studio/

- frontend/
- backend/

Additional directories:

- examples/
- tests/
- docs/

# 6. Core Data Abstractions

## WeightModel

Represents a loaded neural network checkpoint.

Must expose:

- Metadata
- Architecture
- Modules
- Parameters
- Buffers
- Tensor registry
- Backend
- Provenance

## TensorDescriptor

Every tensor must expose:

- Name
- Shape
- Dtype
- Device
- Role
- Module
- Layout
- Trainable status
- Checksum

## Three programming levels

Level 1: Tensor-level programming.

Level 2: Architecture-level programming.

Level 3: Behaviour-level analysis and programming.

Tensor-level programming is the universal foundation.

# 7. Python SDK

The proposed public API:

```python
import weightspace as wsp

model = wsp.Model.load(
    "model.safetensors"
)

print(model.summary())

model.inspect(
    target="layers.0.weight"
)

program = wsp.Program("weight_edit")

program.target(
    "layers.0.weight"
).scale(
    factor=0.95
)

compiled = wsp.compile(
    program,
    model=model
)

candidate = compiled.execute(
    mode="preview"
)

candidate.validate()

candidate.export(
    "modified_model.safetensors"
)
```

All APIs must be implemented and tested before being advertised as functional.

# 8. Weight Operations

Initial operations:

- Scale
- Add
- Subtract
- Multiply
- Normalize
- Clamp
- Slice
- Replace
- Permute
- Mask
- Interpolate
- Low-rank addition

Every operation must define:

- Mathematical semantics
- Input requirements
- Output requirements
- Shape compatibility
- Dtype behaviour
- Broadcasting rules
- Numerical validation

# 9. WSP Programming Language

Create a domain-specific language for weight-space programming.

Example:

```python
model "SimpleMLP" from "model.safetensors"

program "modify_first_layer":

    target "layers.0.weight":

        select rows 0 to 15

        scale factor 0.95

    validate:
        assert shape_unchanged
        assert all_finite

    export "modified_model.safetensors"
```

The DSL must include:

- Model declarations
- Programs
- Tensor selection
- Operations
- Variables
- Assertions
- Evaluation
- Transactions
- Export

The DSL must compile into a typed intermediate representation.

It must not execute unrestricted arbitrary Python code.

# 10. Compiler Architecture

Implement this pipeline:

1. Lexer
2. Parser
3. Abstract Syntax Tree
4. Semantic Analyzer
5. Type Checker
6. Intermediate Representation
7. Optimizer
8. Execution Planner
9. Validator
10. Execution Engine

The compiler must validate operations before execution.

Initial optimizations:

- Operation fusion
- Duplicate operation detection
- Dead operation detection
- Execution planning

# 11. Weight Execution Engine

Implement:

- Tensor selection
- Tensor transformation
- Copy-on-write editing
- Transaction management
- Candidate model generation
- Weight differences
- Rollback
- Commit
- Export

Original model checkpoints must remain unchanged during preview operations.

The execution engine must support reproducible modifications.

# 12. Analysis Engine

Initial capabilities:

- Mean
- Standard deviation
- Minimum
- Maximum
- L1 norm
- L2 norm
- Sparsity
- Weight distributions
- Singular-value analysis
- Tensor similarity
- Activation analysis

Architecture-aware analysis must use adapters.

# 13. Evaluation Engine

Support:

- Original versus candidate comparison
- Accuracy
- Loss
- Latency
- Numerical stability
- Regression tests
- Custom metrics
- Dataset-based evaluation

Never claim that a modification improves a model without empirical evaluation.

# 14. Visual IDE

Build a web-based developer environment.

Main interface panels:

1. Project Explorer
2. Architecture Explorer
3. WSP Code Editor
4. Tensor Inspector
5. Matrix Heatmap
6. Weight Distribution Viewer
7. Weight Diff Viewer
8. Execution Console
9. Evaluation Dashboard
10. AI Assistant

The UI should have a professional developer-tool aesthetic.

Use a dark theme as the initial default, with an option for light mode.

The IDE must communicate with the local Python engine through FastAPI.

# 15. AI Agent

The AI agent should help programmers:

- Understand model architecture.
- Inspect tensor metadata.
- Analyze selected weights.
- Generate WSP programs.
- Propose transformations.
- Validate proposed programs.
- Evaluate candidate models.

The AI agent must use the same public programming API and compiler as human developers.

It must not directly bypass the execution engine.

Use explicit approval for model modifications and external data sharing.

# 16. Safety and Reliability

Mandatory requirements:

- Preserve original model files.
- Use transactional editing.
- Validate tensor dimensions.
- Validate dtype compatibility.
- Check for NaN and infinity.
- Prevent unrestricted execution of user-supplied programs.
- Restrict AI agent permissions.
- Avoid unsafe checkpoint deserialization.
- Maintain operation history.
- Support rollback.
- Record model provenance.
- Keep credentials out of source control.

# 17. Development Milestones

## Milestone 1: WeightSpace Core

- Project structure
- Model abstraction
- Tensor descriptor
- Tensor registry
- Safetensors loader
- PyTorch adapter
- Model inspection
- Unit tests

## Milestone 2: Editing Engine

- Tensor selection
- Mathematical operations
- Copy-on-write
- Transactions
- Rollback
- Weight diffs
- Export

## Milestone 3: Python SDK and Compiler

- Program builder
- Python API
- AST
- Typed IR
- Type checking
- Compiler
- Execution planner

## Milestone 4: WSP DSL

- Lexer
- Parser
- Grammar
- Semantic validation
- DSL-to-IR conversion
- Execution

## Milestone 5: Runtime and Evaluation

- Model inference
- Evaluation datasets
- Baseline comparison
- Candidate comparison
- Evaluation reports

## Milestone 6: Visual IDE

- React application
- Model explorer
- Tensor inspector
- WSP editor
- Weight visualization
- Execution console

## Milestone 7: AI Integration

- Provider abstraction
- Agent tools
- Program proposals
- Validation
- Evaluation
- Approval workflow

# 18. First Demonstration Project

Use a small PyTorch MLP trained on MNIST.

Demonstrate:

1. Train and save the network.
2. Load the checkpoint.
3. Inspect model architecture.
4. Inspect tensor weights.
5. Create a WSP program.
6. Compile the program.
7. Modify selected weights.
8. Preview the changes.
9. Evaluate original and modified models.
10. Export the modified model.

The demonstration must prove that direct weight modifications work without invoking gradient-based training during the modification process.

# 19. Development Collaboration Protocol

ChatGPT responsibilities:

- System architecture
- API design
- Compiler design
- Technical review
- Debugging assistance
- Documentation review
- Development planning

Antigravity responsibilities:

- Repository inspection
- Code implementation
- File management
- Running tests
- Debugging in the local environment
- Frontend development
- Git commits and branches

Both must follow this specification.

Before implementing major features, Antigravity should:

1. Explain its implementation plan.
2. Identify affected files.
3. Identify dependencies.
4. Identify architectural risks.

After implementing:

1. Run tests.
2. Report modified files.
3. Report test results.
4. Document unresolved issues.
5. Update development documentation.

Do not silently change established public APIs.

# 20. Immediate Development Instruction

Start with Milestone 1 only.

First inspect the repository.

Then prepare:

- Repository structure
- Core module interfaces
- Data model definitions
- Implementation plan
- Unit test plan

After reviewing the plan, begin implementing the WeightSpace Core.

Do not implement the full application in one operation.

Build small, testable components and maintain a working project throughout development.

---

**Project principle:**

WeightSpace Studio aims to make neural network weights first-class programmable artifacts, with a unified programming language, compiler, execution engine and visual development environment.
