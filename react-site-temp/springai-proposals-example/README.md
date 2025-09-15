# Project Overview

This project is a React.js application designed to facilitate resume matching and context management. It includes various functionalities such as matching resumes based on user queries, retrieving and setting prompt contexts, and accessing chat history.

## Project Structure

The project is organized as follows:

```
springai-proposals-example
├── react-frontend
│   ├── .vscode
│   │   └── launch.json          # Debug launch configuration for the React application
│   ├── src
│   │   ├── App.js               # Main component managing application state and functionalities
│   │   └── index.js             # Entry point for the React application
│   ├── public
│   │   └── index.html           # Main HTML file serving the React application
│   ├── package.json             # Project metadata and dependencies
│   └── README.md                # Documentation for the React application
└── README.md                    # General information about the entire project
```

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd springai-proposals-example
   ```

2. **Install dependencies**:
   Navigate to the `react-frontend` directory and run:
   ```bash
   npm install
   ```

3. **Run the application**:
   Start the development server:
   ```bash
   npm start
   ```
   The application will be available at `http://localhost:3000`.

## Debugging

To debug the application, use the provided launch configuration in `.vscode/launch.json`. This configuration allows you to launch Chrome against the local server.

## Features

- **Resume Matching**: Users can input queries to match resumes.
- **Context Management**: Users can get and set prompt contexts based on their session IDs.
- **Chat History Retrieval**: Users can access their chat history for reference.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.