import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class VQVAE_bodypart(nn.Module):
    def __init__(self,
                 args,
                 parts_code_nb={},  # numbers of quantizer's embeddings
                 parts_code_dim={},  # dimension of quantizer's embeddings
                 parts_output_dim={},  # dims of encoder's output
                 parts_hidden_dim={},  # actually this is the hidden dimension of the conv net.
                 down_t=3,
                 stride_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        self.parts_code_nb = parts_code_nb
        self.parts_code_dim = parts_code_dim
        self.parts_output_dim = parts_output_dim
        self.parts_hidden_dim = parts_hidden_dim
        self.quantizer_type = args.quantizer

        if args.dataname == 't2m':
            parts_input_dim = {
                'Root': 7,
                'R_Leg': 50,
                'L_Leg': 50,
                'Backbone': 60,
                'R_Arm': 60,
                'L_Arm': 60,
            }

            for name in self.parts_name:
                raw_dim = parts_input_dim[name]
                hidden_dim = parts_hidden_dim[name]
                output_dim = parts_output_dim[name]

                encoder = Encoder(raw_dim, output_dim, down_t, stride_t, hidden_dim, depth, dilation_growth_rate, activation=activation, norm=norm)
                decoder = Decoder(raw_dim, output_dim, down_t, stride_t, hidden_dim, depth, dilation_growth_rate, activation=activation, norm=norm)
                setattr(self, f'enc_{name}', encoder)
                setattr(self, f'dec_{name}', decoder)

                code_dim = parts_code_dim[name]
                # [Warning] code_dim (used in quantizer) must match the output_emb_width
                assert code_dim == output_dim
                nb_code = parts_code_nb[name]

                if args.quantizer == "ema_reset":
                    quantizer = QuantizeEMAReset(nb_code, code_dim, args)
                elif args.quantizer == "orig":
                    quantizer = Quantizer(nb_code, code_dim, 1.0)
                elif args.quantizer == "ema":
                    quantizer = QuantizeEMA(nb_code, code_dim, args)
                elif args.quantizer == "reset":
                    quantizer = QuantizeReset(nb_code, code_dim, args)
                setattr(self, f'quantizer_{name}', quantizer)

        elif args.dataname == 'kit':
            parts_input_dim = {
                'Root': 7,
                'R_Leg': 62,
                'L_Leg': 62,
                'Backbone': 48,
                'R_Arm': 48,
                'L_Arm': 48,
            }

            for name in self.parts_name:
                raw_dim = parts_input_dim[name]
                hidden_dim = parts_hidden_dim[name]
                output_dim = parts_output_dim[name]

                encoder = Encoder(raw_dim, output_dim, down_t, stride_t, hidden_dim, depth, dilation_growth_rate, activation=activation, norm=norm)
                decoder = Decoder(raw_dim, output_dim, down_t, stride_t, hidden_dim, depth, dilation_growth_rate, activation=activation, norm=norm)
                setattr(self, f'enc_{name}', encoder)
                setattr(self, f'dec_{name}', decoder)

                code_dim = parts_code_dim[name]
                # [Warning] code_dim (used in quantizer) must match the output_emb_width
                assert code_dim == output_dim
                nb_code = parts_code_nb[name]

                if args.quantizer == "ema_reset":
                    quantizer = QuantizeEMAReset(nb_code, code_dim, args)
                elif args.quantizer == "orig":
                    quantizer = Quantizer(nb_code, code_dim, 1.0)
                elif args.quantizer == "ema":
                    quantizer = QuantizeEMA(nb_code, code_dim, args)
                elif args.quantizer == "reset":
                    quantizer = QuantizeReset(nb_code, code_dim, args)
                setattr(self, f'quantizer_{name}', quantizer)

        else:
            raise Exception()


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, parts):
        """
        This is used in training transformer (train_t2m_trans.py and the parts ver.),
          for getting the embedding(also named tokens, discrete repre) of motions.

        parts: List, including [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
          Root:     (B, nframes, 7)
          R_Leg:    (B, nframes, 50)
          L_Leg:    (B, nframes, 50)
          Backbone: (B, nframes, 60)
          R_Arm:    (B, nframes, 60)
          L_Arm:    (B, nframes, 60)
        """
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        tokenized_parts = []
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            N, T, _ = x.shape
            # Preprocess
            x_in = self.preprocess(x)  # (B, nframes, in_dim) ==> (B, in_dim, nframes)

            # Encode
            encoder = getattr(self, f'enc_{name}')
            x_encoder = encoder(x_in)  # (B, out_dim, nframes)
            x_encoder = self.postprocess(x_encoder)  # (B, nframes, out_dim)
            x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (B*nframes, out_dim)

            # Quantization
            quantizer = getattr(self, f'quantizer_{name}')
            code_idx = quantizer.quantize(x_encoder)  # (B*nframes, out_dim) --> (B*nframes)
            code_idx = code_idx.view(N, -1)  # (B, nframes)

            tokenized_parts.append(code_idx)

        return tokenized_parts


    def forward(self, parts):
        """
        Forwarding.
        :param parts: List, including [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
          Root:     (B, nframes, 7)
          R_Leg:    (B, nframes, 50)
          L_Leg:    (B, nframes, 50)
          Backbone: (B, nframes, 60)
          R_Arm:    (B, nframes, 60)
          L_Arm:    (B, nframes, 60)
        :return:
        """

        # [Note] remember to be consistent with the self.parts_name when use the x.
        #   self.parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        x_out_list = []
        loss_list = []
        perplexity_list = []
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            # Preprocess
            x_in = self.preprocess(x)  # (B, nframes, in_dim) ==> (B, in_dim, nframes)

            # Encode
            encoder = getattr(self, f'enc_{name}')
            x_encoder = encoder(x_in)

            # Quantization
            quantizer = getattr(self, f'quantizer_{name}')
            x_quantized, loss, perplexity = quantizer(x_encoder)

            # Decoder
            decoder = getattr(self, f'dec_{name}')
            x_decoder = decoder(x_quantized)

            # Postprocess
            x_out = self.postprocess(x_decoder)  # (B, in_dim, nframes) ==> (B, nframes, in_dim)

            x_out_list.append(x_out)
            loss_list.append(loss)
            perplexity_list.append(perplexity)

        # Return the list of x_out, loss, perplexity
        return x_out_list, loss_list, perplexity_list


    def forward_decoder(self, parts):
        """
        This function will be used in evaluation of transformer (eval_bodypart.py).
          It is used to decode the predicted index motion from the transformer.

        Only support BatchSize == 1.

        :param parts: List, including [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
          Root:     (B, codes_len)      B == 1
          R_Leg:    (B, codes_len)      B == 1
          L_Leg:    (B, codes_len)      B == 1
          Backbone: (B, codes_len)      B == 1
          R_Arm:    (B, codes_len)      B == 1
          L_Arm:    (B, codes_len_)     B == 1

          The input parts should have the same codes_len.
          If not, these parts codes should be truncated according to min codes_len before input into this function

        :return:
        """
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        parts_out = []
        base_codes_len = parts[0].shape[1]
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            assert x.shape[0] == 1  # ensure batch size is 1
            codes_len = x.shape[1]
            assert codes_len == base_codes_len  # make sure all parts has same codes_len

            quantizer = getattr(self, f'quantizer_{name}')
            x_d = quantizer.dequantize(x)  # (B, codes_len) => (B, codes_len, code_dim), B == 1

            # It seems the .view() operation does not bring any change.
            #   The code probably is just adapted from the quantizer's code
            x_d = x_d.view(1, codes_len, -1).permute(0, 2, 1).contiguous()  # (B, code_dim, codes_len)

            # decoder
            decoder = getattr(self, f'dec_{name}')
            x_decoder = decoder(x_d)  # (B, raw_motion_dim, seq_len)
            x_out = self.postprocess(x_decoder)  # (B, seq_len, raw_motion_dim)

            parts_out.append(x_out)

        return parts_out


    def forward_decoder_batch(self, parts):
        """
        Decode the quantized motion to raw motion

        Support computation in batch.

        :param parts: List, including [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
          Root:     (B, codes_len)
          R_Leg:    (B, codes_len)
          L_Leg:    (B, codes_len)
          Backbone: (B, codes_len)
          R_Arm:    (B, codes_len)
          L_Arm:    (B, codes_len_)

          The input parts should have the same codes_len.
          If not, these parts codes should be truncated according to min codes_len before input into this function

        :return:
        """
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        parts_out = []
        base_codes_len = parts[0].shape[1]
        for i, name in enumerate(self.parts_name):  # parts_name: ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']

            x = parts[i]
            B = x.shape[0] # ensure batch size is 1
            codes_len = x.shape[1]
            assert codes_len == base_codes_len  # make sure all parts has same codes_len

            quantizer = getattr(self, f'quantizer_{name}')
            x_d = quantizer.dequantize(x)  # (B, codes_len) => (B, codes_len, code_dim), B == 1

            # It seems the .view() operation does not bring any change.
            #   The code probably is just adapted from the quantizer's code
            x_d = x_d.view(B, codes_len, -1).permute(0, 2, 1).contiguous()  # (B, code_dim, codes_len)

            # decoder
            decoder = getattr(self, f'dec_{name}')
            x_decoder = decoder(x_d)  # (B, raw_motion_dim, seq_len)
            x_out = self.postprocess(x_decoder)  # (B, seq_len, raw_motion_dim)

            parts_out.append(x_out)

        return parts_out



class HumanVQVAEBodyPart(nn.Module):
    def __init__(self,
                 args,
                 parts_code_nb={},  # numbers of quantizer's embeddings
                 parts_code_dim={},  # dimension of quantizer's embeddings
                 parts_output_dim={},  # dims of encoder's output
                 parts_hidden_dim={},  # actually this is the hidden dimension of the conv net.
                 down_t=3,
                 stride_t=2,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_bodypart(args, parts_code_nb, parts_code_dim, parts_output_dim, parts_hidden_dim, down_t, stride_t, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        quants = self.vqvae.encode(x)
        return quants

    def forward(self, x):

        x_out_list, loss_list, perplexity_list = self.vqvae(x)

        return x_out_list, loss_list, perplexity_list

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out

    def forward_decoder_batch(self, x):
        x_out = self.vqvae.forward_decoder_batch(x)
        return x_out
        