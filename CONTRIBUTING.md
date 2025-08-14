# Contributing to qt-brbooth

Thank you for contributing to the qt-brbooth project! This guide will help you understand how to contribute effectively.

## Getting Started

### Prerequisites
- Follow the [SETUP.md](SETUP.md) guide to set up your development environment
- Ensure you have Qt 6.5+, OpenCV 4.11, and Visual Studio installed
- Familiarize yourself with the project structure

### Development Workflow

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/qt-brbooth.git
   cd qt-brbooth
   ```

2. **Set Up Upstream**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/qt-brbooth.git
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

4. **Make Your Changes**
   - Follow the coding standards below
   - Test your changes thoroughly
   - Update documentation if needed

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Fill out the PR template
   - Request review from team members

## Coding Standards

### C++ Code Style
- Use **C++17** features
- Follow **Qt coding conventions**
- Use **meaningful variable names**
- Add **proper comments** for complex logic

### File Organization
```
qt-brbooth/
├── main.cpp                    # Application entry point
├── brbooth.cpp                 # Main application window
├── capture.cpp                 # Camera and segmentation logic
├── tflite_deeplabv3.cpp        # Segmentation algorithm
├── tflite_segmentation_widget.cpp # UI controls
├── background.cpp              # Background management
├── foreground.cpp              # Foreground overlay management
├── dynamic.cpp                 # Dynamic content handling
├── final.cpp                   # Final output processing
└── iconhover.cpp               # UI interaction handling
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `TFLiteDeepLabv3`)
- **Functions**: camelCase (e.g., `performSegmentation`)
- **Variables**: camelCase (e.g., `segmentationResult`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_CONFIDENCE`)
- **Files**: lowercase with underscores (e.g., `tflite_deeplabv3.cpp`)

### Code Documentation
```cpp
/**
 * @brief Performs real-time person segmentation on input frame
 * @param inputFrame The input camera frame to process
 * @return Processed frame with segmentation applied
 * 
 * This function uses edge detection and contour analysis to create
 * person silhouettes for background replacement.
 */
cv::Mat TFLiteDeepLabv3::performOpenCVSegmentation(const cv::Mat &inputFrame)
{
    // Implementation here
}
```

### Error Handling
```cpp
try {
    // Your code here
    cv::Mat result = processFrame(input);
    return result;
} catch (const std::exception& e) {
    qWarning() << "Error processing frame:" << e.what();
    return inputFrame.clone(); // Return original frame as fallback
}
```

### Debug Output
```cpp
// Use qDebug() for informational messages
qDebug() << "Segmentation initialized successfully";

// Use qWarning() for warnings
qWarning() << "Low confidence detection:" << confidence;

// Use qCritical() for critical errors
qCritical() << "Failed to load model:" << modelPath;
```

## Testing Guidelines

### Before Submitting
- ✅ **Build Successfully**: Project compiles without errors
- ✅ **Functionality Test**: All features work as expected
- ✅ **Performance Test**: No significant performance regression
- ✅ **Memory Test**: No memory leaks (use Valgrind if available)
- ✅ **Cross-platform**: Test on different configurations

### Testing Checklist
```bash
# 1. Clean build
make clean
qmake qt-brbooth.pro
make

# 2. Run application
./qt-brbooth

# 3. Test features
- Camera opens and works
- Segmentation functions properly
- Background replacement works
- Video recording functions
- UI interactions work
- No crashes or errors

# 4. Performance check
- Real-time processing (30+ FPS)
- Responsive UI
- No memory leaks
```

## Pull Request Guidelines

### PR Title Format
```
Type: Brief description of changes

Examples:
- Feature: Add new background template support
- Bugfix: Fix segmentation edge detection issue
- Performance: Optimize contour filtering algorithm
- Documentation: Update setup instructions
```

### PR Description Template
```markdown
## Description
Brief description of what this PR does.

## Changes Made
- [ ] Added new feature X
- [ ] Fixed bug in Y
- [ ] Updated documentation
- [ ] Performance improvements

## Testing
- [ ] Built successfully
- [ ] Tested on Windows 10/11
- [ ] All features working
- [ ] No performance regression

## Screenshots (if applicable)
Add screenshots showing the changes.

## Related Issues
Closes #123
Fixes #456
```

### Review Process
1. **Self-Review**: Review your own code before submitting
2. **Team Review**: Request review from at least one team member
3. **Address Feedback**: Respond to review comments promptly
4. **Merge**: Once approved, merge to main branch

## Issue Reporting

### Bug Reports
When reporting bugs, include:
- **Description**: What happened vs. what was expected
- **Steps to Reproduce**: Detailed steps to recreate the issue
- **Environment**: OS, Qt version, OpenCV version, hardware
- **Screenshots**: Visual evidence of the issue
- **Logs**: Any error messages or debug output

### Feature Requests
When requesting features, include:
- **Description**: What you want to achieve
- **Use Case**: Why this feature is needed
- **Proposed Solution**: How you think it should work
- **Mockups**: Visual examples if applicable

## Communication

### Team Communication
- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code review and discussion
- **README.md**: For project documentation
- **SETUP.md**: For setup instructions

### Code Review Guidelines
- **Be Constructive**: Provide helpful, specific feedback
- **Ask Questions**: If something is unclear, ask for clarification
- **Suggest Improvements**: Offer specific suggestions for better code
- **Be Respectful**: Remember that code review is a collaborative process

## Release Process

### Version Numbering
- **Major.Minor.Patch** (e.g., 2.1.0)
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version number updated
- [ ] Changelog updated
- [ ] Release notes written
- [ ] Tagged release on GitHub

## Getting Help

### Resources
- [Qt Documentation](https://doc.qt.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [C++ Reference](https://en.cppreference.com/)
- [Git Documentation](https://git-scm.com/doc)

### Asking for Help
1. **Search First**: Check existing issues and documentation
2. **Be Specific**: Provide detailed information about your problem
3. **Include Context**: Share relevant code and error messages
4. **Be Patient**: Team members will respond when available

## Code of Conduct

### Our Standards
- **Be Respectful**: Treat all team members with respect
- **Be Collaborative**: Work together to improve the project
- **Be Professional**: Maintain professional communication
- **Be Inclusive**: Welcome contributions from all team members

### Unacceptable Behavior
- Harassment or discrimination
- Inappropriate or offensive language
- Spam or off-topic discussions
- Disruptive behavior

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to qt-brbooth! Your contributions help make this project better for everyone. 