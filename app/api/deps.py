from fastapi import Depends

def get_dummy_user():
    return {"id": "anon"}
