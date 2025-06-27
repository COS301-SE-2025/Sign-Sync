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
            },

            {
                test: /\.css$/i,
                use: [
                "style-loader",  // inject styles into DOM
                "css-loader",    // turns css into commonjs
                "postcss-loader" // runs Tailwind and autoprefixer
                ],
            },

            { //for images
                test: /\.(png|jpe?g|gif|svg)$/i,
                type: "asset/resource", // this tells Webpack to emit the file and give you a URL
            },

            { //for models
                test: /\.(task|bin|wasm)$/i,
                type: "asset/resource", // this tells Webpack to emit the file and give you a URL
                generator:{
                    filename: 'assets/[name][ext]',
                }
            },

            { // For 3D models imported from Blender
                test: /\.(glb|gltf)$/,
                type: "asset/resource",
                generator: {
                    filename: 'assets/3DModels/[name][ext]',
                },
            }
        ]
        
    },

    devServer: {
        historyApiFallback: true, //so that index.html serves all routes, didn't work without this. 
    }
}