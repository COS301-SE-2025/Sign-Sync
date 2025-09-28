import requests
import pytest
import websockets

ENDPOINT = "https://apigateway-evbsd4dmhbbyhwch.southafricanorth-01.azurewebsites.net/api/"

def test_asl_to_english_translate():
    payload = {"text": "store i go"}

    response = requests.post(ENDPOINT + "word/translate", json=payload)

    assert response.status_code == 200

    response.raise_for_status()
    data = response.json()
    translation = data.get("translation", "")

    assert translation == "I go to the store."

    payload = {"text": "i go house"}

    response = requests.post(ENDPOINT + "word/translate", json=payload)

    assert response.status_code == 200
    response.raise_for_status()
    data = response.json()
    translation = data.get("translation", "")
    assert translation == "I go to the house."

def test_alphabet():
    payload = {
	"keypoints": [{"x":0.3645949363708496,"y":0.7240530252456665,"z":-5.588693738900474e-7},{"x":0.45150837302207947,"y":0.648549497127533,"z":-0.020198583602905273},{"x":0.5101950168609619,"y":0.5146275758743286,"z":-0.0212872251868248},{"x":0.5226097702980042,"y":0.40054386854171753,"z":-0.02596849761903286},{"x":0.5021657943725586,"y":0.3404342234134674,"z":-0.02755439653992653},{"x":0.4547276794910431,"y":0.4225098788738251,"z":0.007961568422615528},{"x":0.470999538898468,"y":0.33623573184013367,"z":-0.03427174314856529},{"x":0.46783646941185,"y":0.42765629291534424,"z":-0.058822453022003174},{"x":0.45790594816207886,"y":0.5172269940376282,"z":-0.06419108062982559},{"x":0.4018481373786926,"y":0.4247318208217621,"z":0.004770350642502308},{"x":0.41737252473831177,"y":0.3362211287021637,"z":-0.04547403007745743},{"x":0.4148767590522766,"y":0.4639118015766144,"z":-0.05938538536429405},{"x":0.4084458649158478,"y":0.5676688551902771,"z":-0.052197568118572235},{"x":0.34891968965530396,"y":0.43481579422950745,"z":-0.006067061331123114},{"x":0.3631158471107483,"y":0.3549724817276001,"z":-0.0547511950135231},{"x":0.3692995309829712,"y":0.48027724027633667,"z":-0.049738962203264236},{"x":0.3685874342918396,"y":0.5752197504043579,"z":-0.02812696248292923},{"x":0.29317042231559753,"y":0.4507398009300232,"z":-0.018771963194012642},{"x":0.30367183685302734,"y":0.3667384386062622,"z":-0.04574451968073845},{"x":0.3175533711910248,"y":0.4371810257434845,"z":-0.03538179025053978},{"x":0.3232554495334625,"y":0.5009985566139221,"z":-0.016991080716252327}]
    }

    response = requests.post(ENDPOINT + "alphabet/predict", json=payload)

    assert response.status_code == 200
    response.raise_for_status()
    data = response.json()
    prediction = data.get("prediction", "")

    assert prediction != None


    

