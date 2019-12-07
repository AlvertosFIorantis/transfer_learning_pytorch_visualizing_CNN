#%%
from google_images_download import google_images_download

# creating object
response = google_images_download.googleimagesdownload()

search_queries = [

    # 'BT hub',
    'BMW cars',
    'BMW series'
]

#%%
def downloadimages(query):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urs is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {"keywords": query,
                 "limit": 100,
                 "print_urls": True}
    try:
        response.download(arguments)

    # Handling File NotFound Error
    except FileNotFoundError:
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit": 100,
                     "print_urls": True,
                     "size": "medium"}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass

#%%
# Run the code to download images
for query in search_queries:
    downloadimages(query)
    print()


# %%
