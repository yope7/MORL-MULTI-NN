import datetime as dt
import pytz

def get_now_jst():
    return dt.datetime.now(pytz.timezone('Asia/Tokyo'))

def get_timestamp():
    now = get_now_jst()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    return timestamp
