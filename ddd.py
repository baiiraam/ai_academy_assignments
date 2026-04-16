# !usr/bin/bash that means shebang

# print(dir(help))

import requests

response = requests.get("https://www.python.org")
print(response.content)
