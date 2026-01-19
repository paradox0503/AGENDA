# coding = utf-8
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch import nn, Tensor

from util.conf import Configuration
from model.ResidualAE import ResidualEncoder, ResidualDecoder, SingleResidualDecoder
from model.DenseAE import DenseEncoder, DenseDecoder
from model.RNNAE import RNNEncoder, RNNDecoder
from model.FDJAE import FDJEncoder, FDJDecoder
from model.InceptionAE import InceptionEncoder, InceptionDecoder
from model.transformer import TEM
from model.transformer import TransformerDecoderModel as TDM
from model.timesnet import TimesNetEncoder, TimesNetDecoder



class AEBuilder(nn.Module):
    def __init__(self, conf: Configuration):
        super(AEBuilder, self).__init__()

        encoder_name = conf.getHP('encoder')
        decoder_name = conf.getHP('decoder')
        self.__decoder =  None

        if encoder_name == 'residual':
            self.__encoder = ResidualEncoder(conf)
            if decoder_name:
                self.__decoder = ResidualDecoder(conf)
        elif encoder_name == 'dense':
            self.__encoder = DenseEncoder(conf)
            if decoder_name:
                self.__decoder = DenseDecoder(conf)
        elif encoder_name == 'fdj':
            self.__encoder = FDJEncoder(conf)
            if decoder_name:
                self.__decoder = FDJDecoder(conf)
        elif encoder_name == 'inception':
            self.__encoder = InceptionEncoder(conf)
            if decoder_name:
                self.__decoder = InceptionDecoder(conf)
        elif encoder_name == 'gru' or encoder_name == 'lstm':
            self.__encoder = RNNEncoder(conf)
            if decoder_name:
                self.__decoder = RNNDecoder(conf)
        elif encoder_name == 'transformer':
            self.__encoder = TEM(conf)
            if decoder_name:
                self.__decoder = TDM(conf)
        elif encoder_name == 'timesnet':
            self.__encoder = TimesNetEncoder(conf)
            if decoder_name:
                self.__decoder = TimesNetDecoder(conf)
        elif encoder_name == 'timesmixer':
            from model.timesmixer import TimesMixerEncoder, TimesMixerDecoder
            self.__encoder = TimesMixerEncoder(conf)
            if decoder_name:
                self.__decoder = TimesMixerDecoder(conf)
        else:
            raise ValueError('encoder {:s} isn\'t supported yet'.format(encoder_name))


    def encode(self, input: Tensor) -> Tensor:
        return self.__encoder(input)


    def decode(self, input: Tensor) -> Tensor:
        if self.__decoder is None:
            raise ValueError('No decoder')

        return self.__decoder(input)


    # explicit model.encode/decode is preferred as decoder might not exist
    # forward is mostly for examining no. parameters
    def forward(self, input: Tensor) -> Tensor:
        embedding = self.encode(input)

        if self.__decoder is None:
            return embedding

        return self.decode(embedding)
