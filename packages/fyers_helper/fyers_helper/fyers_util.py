from fyers_apiv3 import fyersModel
import webbrowser
import base64
import hmac
import os
import struct
import time
from urllib.parse import urlparse, parse_qs
import requests
import os
from retry import retry
import datetime as dt
import pandas as pd


class BaseSession:
    def init_client(self):
        raise NotImplementedError
    
class BaseBroker:
    def buy(self, symbol, qty=1, price=None):
        raise NotImplementedError

    def sell(self, symbol, qty=1, price=None):
        raise NotImplementedError
    
    def positions(self, symbol, product_type):
        raise NotImplementedError
    
    def coverorder(self, symbol, price, sl, qty=1):
        raise NotImplementedError

    def bracketorder(self, symbol, price, sl, tp, direction, qty=1):
        raise NotImplementedError
    
    def bobuy(self, symbol, price, sl, tp, qty=1):
        raise NotImplementedError

    def bosell(self, symbol, price, sl, tp, qty=1):
        raise NotImplementedError
    

class FyersSession(BaseSession):
    def __init__(self, client_id, secret_key, username, totp_key, pin, token=None) -> None:
        self.redirect_uri = (
            "http://127.0.0.1:8080"  ## redircet_uri you entered while creating APP.
        )
        self.grant_type = "authorization_code"  ## The grant_type always has to be "authorization_code"
        self.response_type = "code"  ## The response_type always has to be "code"
        self.state = "sample"  ##  The state field here acts as a session manager. you will be sent with the state field after successfull generation of auth_code

        self.client_id = client_id
        self.secret_key = secret_key
        self.username = username
        self.totp_key = totp_key
        self.pin = pin
        self.token = token
        self.client = None

    def createSessionURL(self):
        appSession = fyersModel.SessionModel(
            self.client_id,
            self.redirect_uri,
            self.response_type,
            state=self.state,
            secret_key=self.secret_key,
            grant_type=self.grant_type,
        )

        # ## Make  a request to generate_authcode object this will return a login url which you need to open in your browser from where you can get the generated auth_code
        generateTokenUrl = appSession.generate_authcode()

        print((generateTokenUrl))
        webbrowser.open(generateTokenUrl, new=1)

    def totp(self, key, time_step=30, digits=6, digest="sha1"):
        key = base64.b32decode(key.upper() + "=" * ((8 - len(key)) % 8))
        counter = struct.pack(">Q", int(time.time() / time_step))
        mac = hmac.new(key, counter, digest).digest()
        offset = mac[-1] & 0x0F
        binary = struct.unpack(">L", mac[offset : offset + 4])[0] & 0x7FFFFFFF
        return str(binary)[-digits:].zfill(digits)

    def init_client(self):
        if self.token is None:
            try:
                self.token = self._get_token()
            except Exception as e:
                raise e
        self.client = fyersModel.FyersModel(
            client_id=self.client_id, token=self.token, log_path=os.getcwd()
        )
        return self.client

    @retry(delay=2, tries=20)
    def _get_token(self):
        headers = {
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        }

        s = requests.Session()
        s.headers.update(headers)

        data1 = f'{{"fy_id":"{base64.b64encode(f"{self.username}".encode()).decode()}","app_id":"2"}}'
        r1 = s.post("https://api-t2.fyers.in/vagator/v2/send_login_otp_v2", data=data1)
        if r1.status_code != 200:
            raise Exception(f"Error in r2:\n {r1.text}")

        request_key = r1.json()["request_key"]
        data2 = f'{{"request_key":"{request_key}","otp":{self.totp(self.totp_key)}}}'
        r2 = s.post("https://api-t2.fyers.in/vagator/v2/verify_otp", data=data2)
        if r2.status_code != 200:
            raise Exception(f"Error in r2:\n {r2.text}")

        request_key = r2.json()["request_key"]
        data3 = f'{{"request_key":"{request_key}","identity_type":"pin","identifier":"{base64.b64encode(f"{self.pin}".encode()).decode()}"}}'
        r3 = s.post("https://api-t2.fyers.in/vagator/v2/verify_pin_v2", data=data3)
        if r3.status_code != 200:
            raise Exception(f"Error in r3:\n {r3.json()}")

        headers = {
            "authorization": f"Bearer {r3.json()['data']['access_token']}",
            "content-type": "application/json; charset=UTF-8",
        }
        data4 = f'{{"fyers_id":"{self.username}","app_id":"{self.client_id[:-4]}","redirect_uri":"{self.redirect_uri}","appType":"100","code_challenge":"","state":"abcdefg","scope":"","nonce":"","response_type":"code","create_cookie":true}}'
        r4 = s.post("https://api-t1.fyers.in/api/v3/token", headers=headers, data=data4)
        if r4.status_code != 308:
            raise Exception(f"Error in r4:\n {r4.json()}")

        parsed = urlparse(r4.json()["Url"])
        auth_code = parse_qs(parsed.query)["auth_code"][0]

        session = fyersModel.SessionModel(
            self.client_id,
            self.redirect_uri,
            self.response_type,
            secret_key=self.secret_key,
            grant_type=self.grant_type,
        )
        session.set_token(auth_code)
        response = session.generate_token()

        return response["access_token"]

class Historical:
    def __init__(self, session: FyersSession) -> None:
        self._session = session
        self._client = self._session.init_client()

    @retry(tries=5, delay=2)
    def historical(self, symbol, resolution, start, end):
        curr = start
        delta = dt.timedelta(days=100) if end - start > dt.timedelta(days=100) else end - start
        final_data = []
        
        while curr < end:
            payload = {
                "symbol": symbol,
                "resolution": f"{resolution}",
                "date_format": "1",
                "range_from": f'{curr:%Y-%m-%d}',
                "range_to": f'{(curr + delta):%Y-%m-%d}',
                "cont_flag": "1",
            }
            data = self._client.history(data=payload)
            time.sleep(0.1)
            try:
                final_data += data["candles"]
            except Exception as e:
                raise e
            curr += delta + dt.timedelta(days=1)
        df = pd.DataFrame(final_data, columns=["datetime", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["datetime"], unit="s", utc=True)
        df.index = df.index.tz_convert("Asia/Kolkata")
        df.datetime = df.index
        df = df.sort_index()
        return df
    
    def buy(self):
        self._client.place_order()
    