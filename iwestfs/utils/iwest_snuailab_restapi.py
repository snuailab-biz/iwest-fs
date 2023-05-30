import requests
import json
from collections import OrderedDict

####################################
# Writer    : DKKO
# Date      : 20230518
# Project   : iwest
####################################


def make_msg_licenseplate(lincense_info:str,bbox:list,timestamp:int,name:str):
    data_temp = OrderedDict();
    data_temp["id"] = "15bbcc33-70d1-4710-a430-ebddb34ea36d";
    data_temp["stream"] = { "app_id":"1ee3564a-7154-4379-a947-d98f2d8b3fd4",
                            "stream_id":"eede9503-c5d4-4185-927b-e75c97ab98ef;a0342c93-b48e-a606-ae1c-679509c08e49",
                            "name":name};
    info = {    "label":"license_plate",
                "bbox" : {"x":bbox[0],"y":bbox[1],"w":bbox[2],"h":bbox[3]},
                "classifiers" : [{ "type":"license_plate","label": lincense_info} ],
                "score":0.98
                };
    
    data_temp["objects"] = [info];

    data_temp["event_type"] = "lpr";
    data_temp["timestamp"] = timestamp;
    data_temp["desc"] = "lpr";
    return data_temp;


def make_msg_flame(bbox_xywh:list,score:float,timestamp:int,name:str):
    data_temp = OrderedDict();
    data_temp["id"] = "bcf45a74-6524-4c42-b604-d53971a10408";
    
    data_temp["stream"] = { "app_id":"1a3a2631-bda1-4a1f-92fe-dbd4dc784efc",
                            "stream_id":"6e03a7c5-e172-7eab-2e44-1b12a48aeed4;a0342c93-b48e-a606-ae1c-679509c08e49",
                            "name":name};
    
    info = {    "label":"flame",
                "bbox" : {"x":bbox_xywh[0],"y":bbox_xywh[1],"w":bbox_xywh[2],"h":bbox_xywh[3]},
                "classifiers" : [],
                "score":str(score)
                };
    
    data_temp["objects"] = [info];

    data_temp["event_type"] = "flame Detection";
    data_temp["timestamp"] = str(timestamp);
    data_temp["desc"] = "flame Detection";
    return data_temp;

def make_msg_smoke(bbox_xywh:list,score:float,timestamp:int,name:str):
    data_temp = OrderedDict();
    data_temp["id"] = "bcf45a74-6524-4c42-b604-d53971a10408";
    
    data_temp["stream"] = { "app_id":"1a3a2631-bda1-4a1f-92fe-dbd4dc784efc",
                            "stream_id":"6e03a7c5-e172-7eab-2e44-1b12a48aeed4;a0342c93-b48e-a606-ae1c-679509c08e49",
                            "name":name};
    
    info = {    "label":"smoke",
                "bbox" : {"x":bbox_xywh[0],"y":bbox_xywh[1],"w":bbox_xywh[2],"h":bbox_xywh[3]},
                "classifiers" : [],
                "score":str(score)
                };
    
    data_temp["objects"] = [info];

    data_temp["event_type"] = "smoke Detection";
    data_temp["timestamp"] = str(timestamp);
    data_temp["desc"] = "smoke Detection";
    return data_temp;


def send_msg_flame(url:str,bbox_xywh:list,score:float,timestamp:int,name:str):
    # headers
    headers = {
        "Content-Type": "application/json"
    }
    temp = make_msg_flame(bbox_xywh,score,timestamp,name);
    data = json.dumps(temp);
    response = requests.post(url, headers=headers, data=data);

def send_msg_smoke(url:str,bbox_xywh:list,score:float,timestamp:int,name:str):
    # headers
    headers = {
        "Content-Type": "application/json"
    }
    temp = make_msg_smoke(bbox_xywh,score,timestamp,name);
    data = json.dumps(temp);
    response = requests.post(url, headers=headers, data=data);


def send_json_msg_licenseplate(url:str,lincense_info:str,bbox:list,timestamp:int):
    # headers
    headers = {
        "Content-Type": "application/json"
    }
    temp = make_msg_licenseplate(lincense_info,bbox,timestamp);
    data = json.dumps(temp);
    response = requests.post(url, headers=headers, data=data);