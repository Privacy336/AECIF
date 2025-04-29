import torch
from torch import nn
from ..captioning_model import CaptioningModel
from models.beam_search import *




class ImageEnhanceTransformer(CaptioningModel):
    def __init__(self, bos_idx, image_enhancement, backbone_1, backbone_2, fusion_model, grid_encoder, encoder, decoder, config):
        super(ImageEnhanceTransformer, self).__init__()

        self.device_a = torch.device('cuda:{}'.format(config.gpu_number[0]))
        self.device_b = torch.device('cuda:{}'.format(config.gpu_number[1]))
        self.device_c = torch.device('cuda:{}'.format(config.gpu_number[2]))
        self.device_d = torch.device('cuda:{}'.format(config.gpu_number[3]))
        self.config = config

        self.bos_idx = bos_idx
        self.image_enhancement_module = image_enhancement.to(self.device_a)
        self.backbone_1 = backbone_1.to(self.device_b)
        self.backbone_2 = backbone_2.to(self.device_c)


        self.fusion_model = fusion_model.to(self.device_d)

        self.grid_encoder = grid_encoder.to(self.device_d)
        self.att_encoder = encoder.to(self.device_d)
        self.decoder = decoder.to(self.device_d)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and not 'backbone' in name:
                nn.init.xavier_uniform_(p)

    def forward(self, mode, img_id, seq=None, max_len=None, eos_idx=None, beam_size=None, out_size=1, return_probs=False):
        images = img_id[0]
        self.image_id = list(img_id[1])

        if mode == 'xe':

            enhanced_images_input = images.to(self.device_a)
            enhanced_images_output = self.image_enhancement_module(enhanced_images_input)
            origial_enc_output_1, origial_enc_output_1_list, _ = self.backbone_1(images.to(self.device_b))
            origial_enc_output_2, origial_enc_output_2_list, _ = self.backbone_2(enhanced_images_output.to(self.device_c))
            origial_enc_output_1_list = [tensor.to(self.device_d) for tensor in origial_enc_output_1_list]
            origial_enc_output_2_list = [tensor.to(self.device_d) for tensor in origial_enc_output_2_list]
            fusion_feature = self.fusion_model(origial_enc_output_1_list, origial_enc_output_2_list)
            grid_feature_1 = self.grid_encoder(fusion_feature)
            enc_output, mask_enc = self.att_encoder(grid_feature_1)
            dec_output = self.decoder(seq.to(self.device_d), enc_output, mask_enc, device=self.device_d)

            return dec_output

        elif mode == 'rl':
            bs = BeamSearch(self, max_len, eos_idx, beam_size, config=self.config)
            return bs.apply(images, out_size, return_probs)

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device), None, None]

    def print_model_parm_nums(model):
        total = sum([param.nelement() for param in model.parameters()])
        print('  + Number of params: %.2fM' % (total / 1e6))

        class myNet(torch.nn.Module):
            def __init__(self, backbone):
                super(myNet, self).__init__()
                self.net = backbone

            def forward(self, x):
                with torch.no_grad():
                    x = self.net(x)
                return x

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None

        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long().to(self.device_b)
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long().to(self.device_b)

                enhanced_images_input = visual.to(self.device_a)
                enhanced_images_output = self.image_enhancement_module(enhanced_images_input).to(self.device_c)

                visual = visual.to(self.device_b)
                self.origial_enc_output_1, self.origial_enc_output_1_list, _  = self.backbone_1(visual)
                self.orgial_enc_output_2,  self.origial_enc_output_2_list, _ = self.backbone_2(enhanced_images_output)

                self.origial_enc_output_1_list = [tensor.to(self.device_d) for tensor in self.origial_enc_output_1_list]
                self.origial_enc_output_2_list = [tensor.to(self.device_d) for tensor in self.origial_enc_output_2_list]
                self.fusion_feature = self.fusion_model(self.origial_enc_output_1_list,
                                                        self.origial_enc_output_2_list)

                self.grid_feature_1 = self.grid_encoder(self.fusion_feature)
                self.enc_output, self.mask_enc = self.att_encoder(self.grid_feature_1)

                if isinstance(visual, torch.Tensor):
                    it = self.enc_output.new_full((self.enc_output.shape[0], 1), self.bos_idx).long().to(self.device_d)

            else:
                it = prev_output.to(self.device_d)

        dec_output = self.decoder(it, self.enc_output, self.mask_enc, device=self.device_d)

        return dec_output, torch.zeros(20, 49).to(self.device_d)



