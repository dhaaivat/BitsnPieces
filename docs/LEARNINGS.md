# Learnings log: BitsnPieces

A running record of what was learnt while preparing and running the precision experiment: concepts, experiment design, and environment issues. Results are in [README.md](README.md). Add new entries at the bottom of the relevant section.

---

## A. Concepts

1. **Exponent bits set range; mantissa bits set precision.**
   - fp16 (5 / 10): precise, but its range is only about 6e-5 to 65,504.
   - bf16 (8 / 7): the full fp32 range, but coarse (eps 7.8e-3).
   - fp32 (8 / 23): both range and precision.
2. **"More bits" doesn't mean "better for every number".** π rounds to 3.140625 in *both* fp16 and bf16. Each format has a fixed grid of representable values, and fp16's extra precision helps some numbers and not others.
3. **The two formats fail differently.**
   - fp16 fails at the extremes: 70,000 becomes inf, 1e-8 becomes 0, and 1e-6 becomes a subnormal with fewer bits.
   - bf16 never runs out of range, but it is approximate everywhere: 70,000 becomes 70,144.
4. **Most modern LLMs are published in bf16.** Qwen2.5 is one. Loading such a model in fp32 only appends zero bits and adds no information. A bf16 round-trip of its weights is lossless, and fp16 is almost lossless because 10 mantissa bits > 7.
5. **Storage precision and compute precision are separate.**
   - Running a model "in fp16" changes both the stored weights and the arithmetic.
   - The `X-weights/fp32-math` controls isolate storage.
   - In this run, *all* observed differences came from compute.
6. **Weights are rarely the fp16 problem; activations are.**
   - The largest weight was 316 and activations peaked around 6.5e3, both under 65,504.
   - Overflow risk lives in the activations (the residual stream and MLP values), not in the weights.
7. **Greedy decoding is chaotic.** A tiny logit change at a near-tie flips one token, and everything after it changes. Identical text is a misleading metric; compare probabilities, KL and task accuracy instead.
8. **Hardware support decides speed.**
   - A T4 (compute capability 7.5) has fp16 tensor cores but no native bf16, so bf16 was *slower than fp32* there.
   - fp16 gains most on long prompts processed in large matrix multiplications: 4.4x on BoolQ vs 1.4x for generation.

## B. Experiment design

1. **Read probabilities instead of generating for yes/no.** Comparing next-token P(yes) vs P(no) is deterministic and gives a continuous metric (ΔP, KL) rather than only right/wrong.
   - "Yes" and "no" each map to several token ids (yes/Yes/␣yes/␣Yes). Add up their probabilities (`logsumexp`) for each side.
2. **Pad batches on the left** so position -1 is always the last real token, for both last-token logits and batched generation.
3. **Run the output layer only on the last position.** Full logits for a batch would be about 8 × 500 tokens × 152k vocab × 4 bytes ≈ 2.4 GB. Calling `m.model(...)` and then `m.lm_head(h[:, -1])` avoids that, and works across transformers versions.
4. **Use controls.** Without the storage-only runs, the fp16 and bf16 differences could not be attributed to storage or to compute.
5. **Blind the judge.** Put all configs' answers to one question in a single call, under shuffled anonymous labels. This prevents bias toward a format and position bias, and keeps it to 50 calls.
6. **Mind the sample size.** On 200 BoolQ questions, one question is 0.5%; on 50 judged answers, one is 2%. Accuracy gaps of that size are noise.
7. **Measure timing noise.** fp32 and the controls do identical math, yet their timings differed by about 7% (generation) and about 25% (BoolQ). Treat speed gaps smaller than that as noise.

## C. Environment and tooling issues (Kaggle, HF, Gemini)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Long pip "dependency conflicts" message on the first cell | `pip install -U google-genai` pulled 3.x; Kaggle's `google-adk` and `aiplatform` need `<3`. The google-colab and bigframes lines were already in the image. | `pip install -q "google-genai>=2.12.1,<3"`. The warnings are not fatal; start a clean session after changing packages. |
| 2 | "You are sending unauthenticated requests to the HF Hub" | The Kaggle secret was read into a Python variable, but `datasets` and `huggingface_hub` read the `HF_TOKEN` *environment variable*. | `os.environ["HF_TOKEN"] = HF_TOKEN` |
| 3 | "`torch_dtype` is deprecated! Use `dtype`" | transformers renamed the argument. | Use `dtype=`, and check the loaded dtype with `assert`, because older versions may silently ignore an unknown argument. |
| 4 | Gemini 404: `gemini-2.5-flash is no longer available to new users` | The model is retired for new keys. | Use `gemini-3.8-flash`, and test it with one call before the loop. List what a key can use with `[m.name for m in client.models.list() if "flash" in m.name]`. |
| 5 | Judge loop kept "retrying" for every question | A failed call silently scores -1 and moves on. | Make the model check fail fast before the loop starts. |
| 6 | GPU memory would never have been freed between configs | `del m` inside a helper function deletes only the function's own name; the caller's `m` keeps the model alive. Caught while writing the notebook. | `del m` in the calling scope, then `gc.collect(); torch.cuda.empty_cache()`. |
| 7 | No result files after the first (interactive) run | Kaggle keeps only files written to `/kaggle/working/` when you **Save Version**, and the save cell hadn't been run. | Save CSVs and PNGs to `/kaggle/working/results/`, then Save Version. **Resolved:** run 2 produced the full set of files. |
| 8 | Summary table appeared as `<pandas.io.formats.style.Styler ...>` when read outside Jupyter | Styled DataFrames render only as HTML. | Read the `text/html` output, or `table.to_csv(...)` it. |
| 9 | Kaggle output came back nested as `results/results/…`, plus `__results___files/` and `__huggingface_repos__.json` | Kaggle's output bundle mirrors `/kaggle/working/`: our `results/` folder sits inside the download folder. It also adds the notebook's inline images (at lower resolution than our `savefig` PNGs) and a record of which HF repos and commits were used. | Use the `savefig` PNGs. `__huggingface_repos__.json` is useful for reproducibility: it pins the exact model and dataset commits. |
| 10 | The first saved `summary.csv` had no BoolQ timing column | `boolq_s` was recorded but not added to the table. | Add it to the table if timing matters (it is only in run 1's printed log for now). |

## D. From the results data

1. **"Save Version" re-runs the whole notebook.** That made run 2 an independent repeat, which gave a free reproducibility check. Every precision metric and divergence point matched run 1, so on a fixed GPU and software stack the numeric effects are deterministic.
2. **An LLM judge can vary between runs even at temperature 0.** The same bf16 answer was scored 0 in run 1 and 1 in run 2, which moved bf16's judge accuracy 0.34 → 0.36. Treat a single-answer judge gap as noise, and check whether scores repeat on a second run.
3. **Check the judge for consistency for free.** Because identical answers are graded in the same call, any identical-text pair with different scores reveals judge noise. There were 0 such cases within a run.
4. **Accuracy flips come from borderline questions.** Both BoolQ flips were on questions fp32 already put within 0.03 of 0.5. bf16 scoring *higher* than fp32 came from one lucky flip, not better math. Look at the size of the ΔP, not only the accuracy.
5. **A "bit-identical" count is a sharp storage test.** The bf16 control matched fp32 on 200/200 P(yes) values; the fp16 control on only 40/200, with differences around 1e-6. That makes the 2,700 flushed weights visible, yet shows they don't matter.
6. **Text divergence did not change correctness.** All 22 bf16 answers that diverged got the same judge score as fp32.

## E. Open questions

- At how many mantissa bits does weight rounding start to hurt BoolQ accuracy?
- Do MLP intermediate activations get closer to 65,504 than the residual stream on long prompts?
- Does a model published in fp32 show real fp16 and bf16 storage effects?
- How does int8 / 4-bit compare with these results?
