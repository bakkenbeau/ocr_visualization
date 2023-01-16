import easyocr

image_option_2 = r'C:\Users\BeauBakken\PycharmProjects\NathanAssignment1\CRAFT_pytorch\test_images\STOP2_sign_letters_only.jpg'


# this needs to run only once to load the model into memory
reader = easyocr.Reader(['en'], gpu=False)

# Instead of the filepath, you can also pass an
# OpenCV image object (numpy array) or an image file as bytes. A URL to a raw image is also acceptable
result = reader.readtext(image_option_2)

print(result)