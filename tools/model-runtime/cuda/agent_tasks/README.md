# agent_tasks — next-session fan-out prompts

Each `.md` file in this directory is a **self-contained Agent prompt**.
The driving conversation dispatches them in parallel via a single
message containing multiple `Agent()` tool calls; each agent lands
a commit on `main` with one kernel or subsystem ported.

These prompts exist because:

1. The kernel ports are independent of each other (each touches its
   own `.cu` file + own `src/kernels/<name>.rs` wrapper).
2. Writing the prompts once and committing them is much cheaper than
   re-explaining context in five separate messages.
3. The prompts themselves are a reviewable artifact: if a reviewer
   disagrees with an instruction, they edit the `.md` before fanout.

## Fan-out order

1. Ship this phase's template (done — see commit introducing
   `rmsnorm.cu`/`rmsnorm.rs`).
2. Read each task's `.md` to confirm scope.
3. In the driving conversation, send **one** message with **five**
   `Agent()` tool-use blocks, each block's `prompt` field containing
   the body of one of these `.md` files.
4. Merge the five agent commits, run `cargo test -p ctox-engine-cuda
   --features cuda --release -- --ignored --nocapture` on the A6000
   to confirm all five kernel tests pass alongside `rmsnorm`.

## Pattern to follow (enforced in every prompt)

* One `.cu` file in `kernels/`, one Rust wrapper module in
  `src/kernels/`.
* Wrapper module name mirrors `.cu` stem.
* Public entry: `launch_<kernel>_<dtype>(device, ...inputs, ...outputs, ...scalars) -> Result<()>`.
* Kernel function cached via `OnceLock<CudaFunction>`.
* No stream sync inside the launch — caller syncs at phase boundary.
* Shape validation before launch, returning `anyhow::Error` on
  mismatch.
* Ignored integration test compares GPU output against a CPU
  reference at Qwen3.5-27B shapes.

Registry (`ptx_registry.rs`) regenerates automatically when new
`.cu` files are added — no manual wiring needed; just commit the
new files.
