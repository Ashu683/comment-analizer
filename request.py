import request
base = 'http://127.0.0.1:5000/'

response = request.post('data':'i am not an idiot')

print(response.json())


