Text embedding comparisons for inventories of Depression (BDI-II), Gaslight, and Dependency

This repository contains surveys and code for embedding the items thereof, and computing inter-survey correlations.

Survey questions can be found in `surveys/`.

All surveys have been flattened such that one line in a document corresponds to the full text of an item.

We employ Qwen3-Embedding (https://github.com/QwenLM/Qwen3-Embedding), and provide an explanation of the
task to the model to aid it in representing the features germane to our objective. For detail, see source. 

To run this comparison, you should have `uv` installed (https://docs.astral.sh/uv/#installation)

In the project root, run

```sh
uv run survey-embeddings
```

If you wish to edit the code (say, to alter the similarity threshold), you may do so by making changes
in `src/survey_embeddings/__init__.py`.

Sample output is provided in [similarity-0.3-qwen3-0.6B.md](/similarity-0.3-qwen3-0.6B.md).

Model citation:

```cite
@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).
