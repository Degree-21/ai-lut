# Project Overview

This project is a Python-based web application that uses the Flask framework to provide a user interface for an AI-powered image style transfer tool. The application takes a user-uploaded image and a desired style preset, and then uses a generative AI model to apply the selected style to the image. It also generates a 3D Look-Up Table (LUT) for color grading, which can be used in video editing software.

The application has two modes of operation:

*   **Web Interface:** A user-friendly web interface for uploading images, selecting styles, and viewing the results.
*   **Command-Line Interface (CLI):** A CLI for processing images in a more automated fashion.

The core logic is implemented in `main.py`, which uses the `Pillow` and `numpy` libraries for image manipulation and color space conversions. The application interacts with a generative AI model via an API to perform the style transfer.

# Building and Running

**1. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**2. Configure the Application:**

Copy `config.example.yaml` to `config.yaml` and fill in the necessary values, including your API key for the generative AI model.

```bash
cp config.example.yaml config.yaml
```

**3. Run the Application:**

*   **Web Interface:**

    ```bash
    python main.py
    ```

    The application will be accessible at `http://127.0.0.1:7860` by default.

*   **Command-Line Interface:**

    To run the CLI, you need to set the `CLI_MODE` environment variable to `1`. You also need to configure the `image_path` in your `config.yaml` file.

    ```bash
    CLI_MODE=1 python main.py
    ```

# Development Conventions

*   **Configuration:** Application configuration is managed through `config.py` and a `config.yaml` file. All sensitive information, such as API keys, should be stored in `config.yaml` and not be committed to version control.
*   **Static Assets:** CSS and JavaScript files are located in the `static` directory.
*   **Templates:** HTML templates are located in the `templates` directory.
*   **Dependencies:** Python dependencies are managed using `pip` and are listed in the `requirements.txt` file.

# Usage

## Web Interface

1.  Start the application by running `python main.py`.
2.  Open your web browser and navigate to `http://127.0.0.1:7860`.
3.  Enter your API key in the provided input field.
4.  Upload an image.
5.  Select one or more style presets.
6.  Click the "Generate" button.
7.  The application will display the generated images and provide download links for the corresponding LUT files.

## Command-Line Interface

1.  Set the `CLI_MODE` environment variable to `1`.
2.  Make sure your `config.yaml` file is properly configured, especially the `image_path` and `api_key` settings.
3.  Run the application: `CLI_MODE=1 python main.py`.
4.  The script will process the image specified in the configuration and save the results in the `outputs` directory.