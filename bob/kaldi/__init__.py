from .cepstral import cepstral
from .dnn import compute_dnn_phone
from .dnn import compute_dnn_vad
from .dnn import nnet_forward
from .gmm import gmm_score
from .gmm import ubm_enroll
from .gmm import ubm_full_train
from .gmm import ubm_train
from .hmm import train_mono

# from .gmm import gmm_score_fast
from .ivector import ivector_extract
from .ivector import ivector_train
from .ivector import plda_enroll
from .ivector import plda_score
from .ivector import plda_train
from .mfcc import compute_vad
from .mfcc import mfcc
from .mfcc import mfcc_from_path


def get_config():
    """Returns a string containing the configuration information.
    """

    import bob.extension

    return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
