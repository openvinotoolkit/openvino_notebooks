var express = require('express');
var proxy = require('http-proxy-middleware');
var logger = require('../logger');

module.exports = function(app, prefix) {
    var router = express.Router();

    router.use(proxy(prefix, {
        target: 'http://127.0.0.1:8081',
        ws: true
    }));

    return router;
}
