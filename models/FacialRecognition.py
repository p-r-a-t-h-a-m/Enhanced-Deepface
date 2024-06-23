from abc import ABC
from typing import Any, Union, List, Tuple
import numpy as np
import torch
from deepface.commons import package_utils

tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model

# Notice that all facial recognition models must be inherited from this class

# pylint: disable=too-few-public-methods
class FacialRecognition(ABC):
    model: Union[Model, Any]
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int

    def forward(self, img: np.ndarray) -> List[float]:
        if isinstance(self.model, Model):
            # TensorFlow/Keras model
            return self.model(img, training=False).numpy()[0].tolist()
        elif isinstance(self.model, torch.nn.Module):
            # PyTorch model
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                img_tensor = torch.from_numpy(img).float()  # Convert numpy array to torch tensor
                if len(img_tensor.shape) == 3:
                    # Add batch dimension if not present
                    img_tensor = img_tensor.unsqueeze(0)
                if img_tensor.shape[-1] == 3:
                    # Convert from (batch, height, width, channels) to (batch, channels, height, width)
                    img_tensor = img_tensor.permute(0, 3, 1, 2)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()  # Move to GPU if available
                    self.model = self.model.cuda()
                embedding = self.model(img_tensor)[0].cpu().numpy().tolist()  # Perform forward pass and convert to list
            return embedding
        else:
            raise ValueError(
                "You must overwrite forward method if it is not a keras or pytorch model,"
                f"but {self.model_name} not overwritten!"
            )












# from abc import ABC
# from typing import Any, Union, List, Tuple
# import numpy as np
# from deepface.commons import package_utils

# tf_version = package_utils.get_tf_major_version()
# if tf_version == 2:
#     from tensorflow.keras.models import Model
# else:
#     from keras.models import Model

# # Notice that all facial recognition models must be inherited from this class

# # pylint: disable=too-few-public-methods
# class FacialRecognition(ABC):
#     model: Union[Model, Any]
#     model_name: str
#     input_shape: Tuple[int, int]
#     output_shape: int

#     def forward(self, img: np.ndarray) -> List[float]:
#         if not isinstance(self.model, Model):
#             raise ValueError(
#                 "You must overwrite forward method if it is not a keras model,"
#                 f"but {self.model_name} not overwritten!"
#             )
#         # model.predict causes memory issue when it is called in a for loop
#         # embedding = model.predict(img, verbose=0)[0].tolist()
#         return self.model(img, training=False).numpy()[0].tolist()
