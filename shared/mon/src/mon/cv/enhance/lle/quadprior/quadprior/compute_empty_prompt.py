# This is how we compute the empty embedding
# You may need to download 'openai/clip-vit-large-patch14'

import pickle

from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from mon.core import Path

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]

model        = FrozenCLIPEmbedder().to("cuda")
embedding    = model.encode([""]).cpu()

print(embedding)
print(embedding.shape)

with open(str(root_dir / "empty_embedding.pkl"), "wb") as f:
    pickle.dump(embedding, f)
