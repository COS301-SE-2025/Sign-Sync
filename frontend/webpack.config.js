//u23524121 Jamean Groenewald

const path = require("path");

module.exports = {
    
    entry: "./frontend/src/index.js",
    
    output: {
        path: path.resolve(__dirname, 'frontend', 'public'),
        filename: 'bundle.js'
    },

    mode: "development",

    module: {
        
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: "babel-loader"
                }
            }
        ]
        
    },

    devServer: {
        historyApiFallback: true, //so that index.html serves all routes, didn't work without this. 
    }
}