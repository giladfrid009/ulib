import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from ulib.pert_module import PertModule


def display_pert(pert_module: PertModule) -> AxesImage:
    """
    Display the perturbation as an image.

    Args:
        pert_module (PertModule): Perturbation to display.

    Returns:
        AxesImage: Plot of the perturbation.
    """

    p = pert_module.to_image().numpy()
    ax = plt.imshow(p)
    plt.axis("off")
    return ax
