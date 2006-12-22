#ifndef GLCANVAS_H
#define GLCANVAS_H
#include <wx/wx.h>
#include <wx/glcanvas.h>
#include "mediator.h"
#include "visualizer.h"
#include "utils.h"

class GLCanvas: public wxGLCanvas {
  public:
    GLCanvas(Mediator* owner,wxWindow* parent,const wxSize &size=wxDefaultSize,
        int* attribList=NULL);
    ~GLCanvas();
    void      display(bool coll_caller=false);
    void      enableDisplay();
    void      disableDisplay();
    Utils::RGB_Color getBackgroundColor() const;
    Utils::RGB_Color getDefaultBackgroundColor() const;
    void      getMaxViewportDims(int *w,int *h);
    unsigned char* getPictureData(int res_x,int res_y);
    void      initialize();
    void      resetView();
    void      reshape();
    void      setActiveTool(int t);
    void      setBackgroundColor(Utils::RGB_Color c);
    void      setDefaultPosition(float structWidth,float structHeight);
    void      setDisplayBackpointers(bool b);
    void      setDisplayStates(bool b);
    void      setDisplayTransitions(bool b);
    void      setDisplayWireframe(bool b);
    void      setVisualizer(Visualizer *vis);

    void      onMouseDown(wxMouseEvent& event);
    void      onMouseEnter(wxMouseEvent& event);
    void      onMouseMove(wxMouseEvent& event);
    void      onMouseUp(wxMouseEvent& event);
    void      onMouseWheel(wxMouseEvent& event);
    void      onPaint(wxPaintEvent& event);
    void      onSize(wxSizeEvent& event);
    void      OnEraseBackground(wxEraseEvent& event);

  private:
    int	      activeTool;
    float     angleX;
    float     angleY;
    int	      currentTool;
    Utils::RGB_Color defaultBGColor;
    bool      displayBackpointers;
    bool      displayStates;
    bool      displayTransitions;
    bool      displayWireframe;
    float     startPosZ;
    float     startPosZDefault;
    bool      collectingData;
    bool      displayAllowed;
    float     farPlane;
    float     farPlaneDefault;
    bool      lightRenderMode;
    Mediator* mediator;
    Utils::Point3D moveVector;
    float     nearPlane;
    int	      oldMouseX;
    int	      oldMouseY;
    Visualizer *visualizer;
    
    void      determineCurrentTool(wxMouseEvent& event);
    void      setMouseCursor();

    DECLARE_EVENT_TABLE()
};

#endif
