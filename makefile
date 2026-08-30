COMPILER_ARGS = -g -mconsole
OBJECT_COMPILER_ARGS = -c $(COMPILER_ARGS)
OBJECTS_FOLDER = build/target/objects
TEST_RESOURCES = tst/resources
TEST_RESOURCES_OUTPUT = build/test/resources
OBJECTS = $(OBJECTS_FOLDER)/Matrix.o $(OBJECTS_FOLDER)/Vector.o $(OBJECTS_FOLDER)/Assert.o $(OBJECTS_FOLDER)/Resources.o $(OBJECTS_FOLDER)/Test_Mat.o $(OBJECTS_FOLDER)/Test_Vector.o $(OBJECTS_FOLDER)/Relu.o $(OBJECTS_FOLDER)/Sigmoid.o $(OBJECTS_FOLDER)/Softmax.o $(OBJECTS_FOLDER)/Test_Relu.o $(OBJECTS_FOLDER)/Test_Sigmoid.o $(OBJECTS_FOLDER)/Test_Softmax.o $(OBJECTS_FOLDER)/Activation_Suite.o $(OBJECTS_FOLDER)/Linal_Suite.o $(OBJECTS_FOLDER)/Gaussian.o $(OBJECTS_FOLDER)/Test_Gaussian.o $(OBJECTS_FOLDER)/Gaussian_Suite.o $(OBJECTS_FOLDER)/Basic_Suite.o $(OBJECTS_FOLDER)/Test_Layer.o $(OBJECTS_FOLDER)/Layer.o $(OBJECTS_FOLDER)/Network.o $(OBJECTS_FOLDER)/Test_Network.o

tst/src/Testing.h tst/src/linal/linal_test.h: src/linal/linal.h

tst/src/activation/activation_test.h: src/activation/activation.h

src\multilayer_perceptron\basic\basic.h: src/linal/linal.h

$(OBJECTS_FOLDER)/Matrix.o: src/linal/Matrix.c src/linal/linal.h
	gcc $< -o $@  $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Vector.o: src/linal/Vector.c src/linal/linal.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Relu.o: src/activation/Relu.c src/activation/activation.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Sigmoid.o: src/activation/Sigmoid.c src/activation/activation.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Softmax.o: src/activation/Softmax.c src/activation/activation.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Gaussian.o: src/stat/Gaussian.c src/stat/gaussian.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Layer.o: src\multilayer_perceptron\basic\Layer.c src\multilayer_perceptron\basic\basic.h src/linal/linal.h src\activation\activation.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Network.o: src\multilayer_perceptron\basic\Network.c src\multilayer_perceptron\basic\basic.h src\linal\linal.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Assert.o: tst/src/Assert.c tst/src/Testing.h src/linal/linal.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Resources.o: tst/src/Resources.c tst/src/Testing.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Runner.o: tst/src/Runner.c tst/src/Testing.h tst/src/linal/linal_test.h tst/src/activation/activation_test.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Gen_Distribution.o: tst/src/Gen_Distribution.c src/stat/gaussian.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Test_Mat.o: tst/src/linal/Test_Mat.c tst/src/linal/linal_test.h src/linal/linal.h tst/src/Testing.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Test_Vector.o: tst/src/linal/Test_Vector.c tst/src/linal/linal_test.h src/linal/linal.h tst/src/Testing.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Test_Relu.o: tst/src/activation/Test_Relu.c tst/src/activation/activation_test.h src/activation/activation.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Test_Sigmoid.o: tst/src/activation/Test_Sigmoid.c tst/src/activation/activation_test.h src/activation/activation.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Test_Softmax.o: tst/src/activation/Test_Softmax.c tst/src/activation/activation_test.h src/activation/activation.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Activation_Suite.o: tst/src/activation/Activation_Suite.c tst/src/activation/activation_test.h tst/src/Testing.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Test_Gaussian.o: tst/src/stat/Test_Gaussian.c tst/src/stat/test_gaussian.h src/stat/gaussian.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Test_Layer.o: tst\src\multilayer_perceptron\basic\Test_Layer.c tst\src\multilayer_perceptron\basic\test_basic.h tst\src\Testing.h src\linal\linal.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Test_Network.o: tst\src\multilayer_perceptron\basic\Test_Network.c tst\src\multilayer_perceptron\basic\test_basic.h src\linal\linal.h src\multilayer_perceptron\basic\basic.h tst\src\Testing.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)
	
$(OBJECTS_FOLDER)/Gaussian_Suite.o: tst/src/stat/Gaussian_Suite.c tst/src/stat/test_gaussian.h tst/src/Testing.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Linal_Suite.o: tst/src/linal/Linal_Suite.c tst/src/linal/linal_test.h tst/src/Testing.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

$(OBJECTS_FOLDER)/Basic_Suite.o: tst\src\multilayer_perceptron\basic\Basic_Suite.c tst\src\multilayer_perceptron\basic\test_basic.h
	gcc $< -o $@ $(OBJECT_COMPILER_ARGS)

build/test/test.exe: $(OBJECTS) $(OBJECTS_FOLDER)/Runner.o
	gcc $^ -o $@ $(COMPILER_ARGS) -lm

build/test/gen_dist.exe: $(OBJECTS) $(OBJECTS_FOLDER)/Gen_Distribution.o
	gcc $^ -o $@ $(COMPILER_ARGS) -lm

test: build/test/test.exe
	$<

gen_dist: build/test/gen_dist.exe
	$<

clean:
	python3 clean.py ./build .o .exe