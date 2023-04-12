import io

import torch
import logging
import transformers
import os

from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

from model import enetv2

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

kernel_type = 'model'
enet_type = 'efficientnet-b7'
out_dim = 9


class ModelHandler(BaseHandler):

    def initialize(self, context):
        """Initialize function loads the model and the tokenizer
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")

        logger.info(f'Properties: {properties}')
        logger.info(f'Manifest: {self.manifest}')

        self.device = torch.device('cpu')
        # load the model
        model_file = self.manifest["model"].get("modelFile", "")
        model_path = os.path.join(model_dir, model_file)

        if model_file:
            self.model = enetv2(enet_type, n_meta_features=0, out_dim=out_dim)
            self.model = self.model.to(self.device)
            state_dict = torch.load(model_file, map_location=self.device)
            state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            logger.info(f'Successfully loaded model from {model_file}')
        else:
            raise RuntimeError('Missing the model file')

        self.initialized = True

    def preprocess(self, requests):

        data = requests[0]

        photo = data['file']
        logger.info(f'Received photo')

        my_transforms = transforms.Compose([transforms.Resize(640),
                                            transforms.CenterCrop(640),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(photo))

        return my_transforms(image)

    def inference(self, inputs):
        outputs = self.model(inputs.unsqueeze(0).to(self.device))
        probabilities = outputs.softmax(1)
        mel_prob, nv_prob = probabilities.data[0][6], probabilities.data[0][7]
        logger.info('Predictions successfully created.')

        return mel_prob.item(), nv_prob.item()

    def postprocess(self, outputs):
        predictions = {'melanoma': outputs[0], 'nevus': outputs[1]}
        logger.info(f'PREDICTED LABELS: {predictions}')

        return [predictions]