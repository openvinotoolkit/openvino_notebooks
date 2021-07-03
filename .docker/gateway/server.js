var express = require('express');
var proxy = require('http-proxy-middleware');
var path = require('path');
var url = require('url');
var fs = require('fs');
var morgan = require('morgan')
var logger = require('./logger');

// Setup the root application. Everything will actually be under a
// mount point corresponding to the specific user. This is added in
// each of the routes when defined.

var app = express();

var uri_root_path = process.env.URI_ROOT_PATH || '';

// Add logging for request.

var log_format = process.env.LOG_FORMAT || 'dev';

app.use(morgan(log_format));

// In OpenShift we are always behind a proxy, so trust the headers sent.

app.set('trust proxy', true);

// Setup handlers for routes.

function install_routes(directory) {
    if (fs.existsSync(directory)) {
        var files = fs.readdirSync(directory);

        for (var i=0; i<files.length; i++) {
            var filename = files[i];

            if (filename.endsWith('.js')) {
                var basename = filename.split('.').slice(0, -1).join('.');

                if (basename != 'index') {
                    var prefix = uri_root_path + '/' + basename;

                    app.get('^' + prefix + '$', function (req, res) {
                        res.redirect(url.parse(req.url).pathname + '/');
                    });

                    var pathname = path.join(directory, filename);
                    var router = require(pathname)(app, prefix + '/');

                    logger.info('Install route for', {path:pathname});

                    app.use(prefix + '/', router);
                }
            }
        }
    }
}

install_routes(path.join(__dirname, 'routes'));

app.use(proxy(uri_root_path, {
    target: 'http://127.0.0.1:8082',
    ws: true
}));

// Start the listener.

logger.info('Start listener');

app.listen(8080);
