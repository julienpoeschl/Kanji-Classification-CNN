"""
Not ment to be imported!

Run this file to quickly check if pytorch is using cuda and the target device was found.
"""

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")