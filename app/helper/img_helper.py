import base64


def save_base64_image(base64_string, path_to_image):
    decoded_image_data = base64.b64decode(base64_string)
    with open(path_to_image, "wb") as image_file:
        image_file.write(decoded_image_data)
