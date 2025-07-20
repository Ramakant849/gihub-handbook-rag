#!/usr/bin/env python3
import os
import json
import logging
import requests
import hashlib
import base64
import abc
import typing as t
import time
import threading
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from Crypto.Cipher import AES

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "lark_bot.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lark_bot")

APP_ID = os.getenv("LARK_APP_ID")
APP_SECRET = os.getenv("LARK_APP_SECRET")
VERIFICATION_TOKEN = os.getenv("LARK_VERIFICATION_TOKEN")
ENCRYPT_KEY = os.getenv("LARK_ENCRYPT_KEY")
LARK_HOST = os.getenv("LARK_HOST", "https://open.feishu.cn")

TENANT_ACCESS_TOKEN_URI = "/open-apis/auth/v3/tenant_access_token/internal"
MESSAGE_URI = "/open-apis/im/v1/messages"

PROCESSED_MESSAGES_FILE = os.path.join(PROJECT_ROOT, "logs", "processed_messages.txt")
os.makedirs(os.path.dirname(PROCESSED_MESSAGES_FILE), exist_ok=True)
if not os.path.exists(PROCESSED_MESSAGES_FILE):
    open(PROCESSED_MESSAGES_FILE, 'a').close()

def is_message_processed(message_id):
    """Check if a message has already been processed"""
    try:
        with open(PROCESSED_MESSAGES_FILE, 'r') as f:
            processed_ids = [line.strip() for line in f.readlines()]
            return message_id in processed_ids
    except Exception as e:
        logger.error(f"Error checking processed messages: {str(e)}")
        return False

def mark_message_processed(message_id):
    """Mark a message as processed"""
    try:
        with open(PROCESSED_MESSAGES_FILE, 'a') as f:
            f.write(f"{message_id}\n")
        cleanup_processed_messages()
    except Exception as e:
        logger.error(f"Error marking message as processed: {str(e)}")

def cleanup_processed_messages():
    """Keep only the most recent 1000 message IDs"""
    try:
        if os.path.getsize(PROCESSED_MESSAGES_FILE) > 100 * 1024:  # 100KB
            with open(PROCESSED_MESSAGES_FILE, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 1000:
                with open(PROCESSED_MESSAGES_FILE, 'w') as f:
                    f.writelines(lines[-1000:])
                logger.info(f"Cleaned up processed messages file, kept {min(1000, len(lines))} entries")
    except Exception as e:
        logger.error(f"Error cleaning up processed messages: {str(e)}")

class Obj(dict):
    """Convert dict to object for easier attribute access"""
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Obj(b) if isinstance(b, dict) else b)

def dict_2_obj(d: dict):
    """Convert dictionary to object"""
    return Obj(d)

class AESCipher(object):
    """AES encryption/decryption for Lark messages"""
    def __init__(self, key):
        self.bs = AES.block_size
        self.key = hashlib.sha256(AESCipher.str_to_bytes(key)).digest()

    @staticmethod
    def str_to_bytes(data):
        u_type = type(b"".decode("utf8"))
        if isinstance(data, u_type):
            return data.encode("utf8")
        return data

    @staticmethod
    def _unpad(s):
        return s[: -ord(s[len(s) - 1 :])]

    def decrypt(self, enc):
        iv = enc[: AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size :]))

    def decrypt_string(self, enc):
        enc = base64.b64decode(enc)
        return self.decrypt(enc).decode("utf8")

class LarkException(Exception):
    """Exception for Lark API errors"""
    def __init__(self, code=0, msg=None):
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        return "{}:{}".format(self.code, self.msg)

    __repr__ = __str__

class MessageApiClient(object):
    """Client for sending messages to Lark"""
    def __init__(self, app_id, app_secret, lark_host):
        self._app_id = app_id
        self._app_secret = app_secret
        self._lark_host = lark_host
        self._tenant_access_token = ""

    @property
    def tenant_access_token(self):
        return self._tenant_access_token

    def send_text_with_open_id(self, open_id, content):
        """Send text message to user by open_id"""
        self.send("open_id", open_id, "text", content)

    def send(self, receive_id_type, receive_id, msg_type, content):
        """Send message to user"""
        self._authorize_tenant_access_token()
        url = "{}{}?receive_id_type={}".format(
            self._lark_host, MESSAGE_URI, receive_id_type
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.tenant_access_token,
        }

        req_body = {
            "receive_id": receive_id,
            "content": content,
            "msg_type": msg_type,
        }
        resp = requests.post(url=url, headers=headers, json=req_body)
        MessageApiClient._check_error_response(resp)

    def _authorize_tenant_access_token(self):
        """Get tenant_access_token and set it for requests"""
        url = "{}{}".format(self._lark_host, TENANT_ACCESS_TOKEN_URI)
        req_body = {"app_id": self._app_id, "app_secret": self._app_secret}
        response = requests.post(url, req_body)
        MessageApiClient._check_error_response(response)
        self._tenant_access_token = response.json().get("tenant_access_token")

    @staticmethod
    def _check_error_response(resp):
        """Check if the response contains error information"""
        if resp.status_code != 200:
            resp.raise_for_status()
        response_dict = resp.json()
        code = response_dict.get("code", -1)
        if code != 0:
            logging.error(response_dict)
            raise LarkException(code=code, msg=response_dict.get("msg"))

class InvalidEventException(Exception):
    """Exception for invalid events"""
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self) -> str:
        return "Invalid event: {}".format(self.error_info)

    __repr__ = __str__

class Event(object):
    """Base class for events"""
    callback_handler = None

    def __init__(self, dict_data, token, encrypt_key):
        header = dict_data.get("header")
        event = dict_data.get("event")
        if header is None or event is None:
            raise InvalidEventException("request is not callback event(v2)")
        self.header = dict_2_obj(header)
        self.event = dict_2_obj(event)
        self._validate(token, encrypt_key)

    def _validate(self, token, encrypt_key):
        """Validate event signature"""
        if self.header.token != token:
            raise InvalidEventException("invalid token")
        timestamp = request.headers.get("X-Lark-Request-Timestamp")
        nonce = request.headers.get("X-Lark-Request-Nonce")
        signature = request.headers.get("X-Lark-Signature")
        body = request.data
        bytes_b1 = (timestamp + nonce + encrypt_key).encode("utf-8")
        bytes_b = bytes_b1 + body
        h = hashlib.sha256(bytes_b)
        if signature != h.hexdigest():
            raise InvalidEventException("invalid signature in event")

    @abc.abstractmethod
    def event_type(self):
        return self.header.event_type

class MessageReceiveEvent(Event):
    """Event for received messages"""
    @staticmethod
    def event_type():
        return "im.message.receive_v1"

class UrlVerificationEvent(Event):
    """Event for URL verification"""
    def __init__(self, dict_data):
        self.event = dict_2_obj(dict_data)

    @staticmethod
    def event_type():
        return "url_verification"

class EventManager(object):
    """Manager for handling events"""
    event_callback_map = dict()
    event_type_map = dict()
    _event_list = [MessageReceiveEvent, UrlVerificationEvent]

    def __init__(self):
        for event in EventManager._event_list:
            EventManager.event_type_map[event.event_type()] = event

    def register(self, event_type: str) -> t.Callable:
        """Register a handler for an event type"""
        def decorator(f: t.Callable) -> t.Callable:
            self.register_handler_with_event_type(event_type=event_type, handler=f)
            return f
        return decorator

    @staticmethod
    def register_handler_with_event_type(event_type, handler):
        EventManager.event_callback_map[event_type] = handler

    @staticmethod
    def get_handler_with_event(token, encrypt_key):
        """Get the handler and event object for a request"""
        dict_data = json.loads(request.data)
        dict_data = EventManager._decrypt_data(encrypt_key, dict_data)
        callback_type = dict_data.get("type")
        
        if callback_type == "url_verification":
            event = UrlVerificationEvent(dict_data)
            return EventManager.event_callback_map.get(event.event_type()), event

        schema = dict_data.get("schema")
        if schema is None:
            raise InvalidEventException("request is not callback event(v2)")

        event_type = dict_data.get("header").get("event_type")
        event = EventManager.event_type_map.get(event_type)(dict_data, token, encrypt_key)
        return EventManager.event_callback_map.get(event_type), event

    @staticmethod
    def _decrypt_data(encrypt_key, data):
        """Decrypt data if needed"""
        encrypt_data = data.get("encrypt")
        if encrypt_key == "" and encrypt_data is None:
            return data
        if encrypt_key == "":
            raise Exception("ENCRYPT_KEY is necessary")
        cipher = AESCipher(encrypt_key)
        return json.loads(cipher.decrypt_string(encrypt_data))

message_api_client = MessageApiClient(APP_ID, APP_SECRET, LARK_HOST)
event_manager = EventManager()

import logging

logger = logging.getLogger("lark_handler")

def init_lark_bot(app):
    """Placeholder for initializing the Lark bot handler."""
    logger.info("[LarkHandler] Initializing Lark bot.")
    # In a real scenario, this would set up Lark bot routes and logic
    pass

