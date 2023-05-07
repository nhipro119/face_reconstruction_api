from application import application 
app = application.Application()
mesh = app.execute_predict("./CTM05853_0.obj")
mesh.export("./out.obj")