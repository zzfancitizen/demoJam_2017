import numpy as np
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import simplejson as json

sys.path.append(os.path.abspath('./'))
from model_restore import model_read


class predict_server(BaseHTTPRequestHandler):
    def do_GET(self):
        request_path = self.path

        print("\n----- Request Start ----->\n")
        print(request_path)
        print(self.headers)
        print("<----- Request End -----\n")

        self.send_response(200)
        self.send_header("Set-Cookie", "foo=bar")
        self.end_headers()

    def do_POST(self):

        predict_model = model_read()

        request_path = self.path
        o = urlparse(request_path)
        if o.path == '/predict':

            content_len = int(self.headers.get_all('content-length', 0)[0])
            post_body = self.rfile.read(content_len)
            test_data = json.loads(post_body)
            temp = np.reshape(test_data["tempm"], (-1, 1))

            predict_model.temp = temp
            breakpoint = predict_model.predict()

            self.send_response(200)
            self.end_headers()

            data = {}
            data["possibility"] = str(breakpoint[0, 0])
            json_data = json.dumps(data)

            self.wfile.write(json_data.encode())
        else:
            self.send_response(404)
            self.end_headers()

    do_PUT = do_POST
    do_DELETE = do_GET


def main():
    PORT = 50030
    print('Listening on localhost:%s' % PORT)
    server = HTTPServer(('', PORT), predict_server)
    server.serve_forever()


if __name__ == '__main__':
    main()
