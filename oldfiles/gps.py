'''
import geocoder

# Obtenez l'adresse IP publique du Raspberry Pi
ip = geocoder.ip('me')
print(ip)
# Obtenez la localisation à partir de l'adresse IP
location = geocoder.osm(ip.ip)

# Imprimez le nom, la longitude et la latitude
print(location)
print(location.city)
print(location.lng)
print(location.lat)
import requests

# Obtenez l'adresse IP publique du Raspberry Pi
ip = requests.get('https://api.ipify.org').text

# Utilisez le service ipinfo.io pour obtenir la localisation
response = requests.get(f'https://ipinfo.io/{ip}/json')

# Obtenez la latitude et la longitude
data = response.json()
print(data)
latitude, longitude = data['loc'].split(',')

# Imprimez la latitude et la longitude
print(latitude)
print(longitude)

'''

import requests

# Obtient l'adresse IP de votre Raspberry Pi
ip_address = requests.get('https://api.ipify.org').text

# Envoie une requête à l'API GeoIP
url = 'http://ip-api.com/json/' + ip_address
response = requests.get(url)

# Analyse la réponse de l'API
if response.status_code == 200:
    location = response.json()
    print("Pays:", location['country'])
    print("Ville:", location['city'])
    print("Latitude:", location['lat'])
    print("Longitude:", location['lon'])
else:
    print("Impossible d'obtenir votre localisation.")
