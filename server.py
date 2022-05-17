import uvicorn

if __name__ == '__main__':
    uvicorn.run("main:app",
                host="192.168.1.232",
                port=8000,
                reload=True,
                # ssl_keyfile="./key.pem", 
                # ssl_certfile="./cert.pem"
                )