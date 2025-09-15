# React Application

This project is a React.js application designed to manage resume matching and chat history functionalities. Below is an overview of the project's structure and its components.

## Project Structure

```
springai-proposals-example
├── react-frontend
│   ├── .vscode
│   │   └── launch.json
│   ├── src
│   │   ├── App.js
│   │   └── index.js
│   ├── public
│   │   └── index.html
│   ├── package.json
│   └── README.md
└── README.md
```

## Components

- **App.js**: The main component that handles state management for session ID, loading states, and various functionalities such as matching resumes, getting and setting prompt context, and retrieving chat history.

- **index.js**: The entry point of the application that renders the `App` component into the DOM.

- **index.html**: The main HTML file that serves the React application, including a root div for mounting the React app.

- **package.json**: Contains metadata about the project, including dependencies and scripts for npm.

## Getting Started

To run the application, follow these steps:

1. Clone the repository.
2. Navigate to the `react-frontend` directory.
3. Install the dependencies using npm:
   ```
   npm install
   ```
4. Start the development server:
   ```
   npm start
   ```
5. Open your browser and navigate to `http://localhost:3000` to view the application.

## Debugging

To debug the application, use the provided launch configuration in the `.vscode/launch.json` file. This configuration allows you to launch Chrome against the localhost server.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.