import transfer_image
import transfer_bottleneck

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def main(argv=None):
    image_lists = transfer_image.create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

