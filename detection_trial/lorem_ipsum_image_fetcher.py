import requests
from pathlib import Path
import uuid

def download_random_image(width, height, save_path):
    """
    Downloads a random image from picsum.photos with the specified width and height.

    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :param save_path: The local path where the image will be saved.
    """
    url = f'https://picsum.photos/{width}/{height}'
    try:
        # Send GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Save the image content to a file
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f'Image saved to {save_path}')

    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')

# Example usage:
if __name__ == '__main__':
    number_of_images = 95
    width = 200
    height = 200
    for _ in range(number_of_images):
        save_path = Path(__file__).resolve().parent / 'src_image' / 'lorem_ipsums' / f"{uuid.uuid4()}.png"
        download_random_image(width, height, save_path)

