import React, {useState, useRef, useEffect} from 'react';
import {
  SafeAreaView,
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import Svg, {Line, Rect, Path} from 'react-native-svg';
import {startSpeechToText} from 'react-native-voice-to-text';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';

const App = () => {
  const [micActive, setMicActive] = useState(false);
  const [messages, setMessages] = useState([]);
  const {hasPermission, requestPermission} = useCameraPermission();
  const device = useCameraDevice('back');
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission, device]);

  const handleMicPress = async () => {
    setMicActive(true);
    try {
      const audioText = await startSpeechToText();
      if (audioText && audioText.trim() !== '') {
        setMessages(prev => [...prev, {text: audioText, from: 'user'}]);
      }
    } catch (error) {
      console.log(error);
    }
    setMicActive(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.topBar}>
        <Text style={styles.title}>HearSay</Text>
      </View>
      <View style={styles.cameraView}>
        {device != null && hasPermission ? (
          <Camera
            style={styles.camera}
            device={device}
            isActive={true}
            video={true}
            photo={true}
          />
        ) : (
          <View style={styles.cameraPlaceholder}>
            <Text style={styles.placeholderText}>
              {!hasPermission
                ? 'Camera permission required'
                : 'No camera device found'}
            </Text>
          </View>
        )}
      </View>
      <ScrollView
        style={styles.chatContainer}
        contentContainerStyle={{padding: 10}}>
        {messages.map((msg, idx) => (
          <View key={idx} style={styles.msgRow}>
            <View style={styles.bubble}>
              <Text style={styles.bubbleText}>{msg.text}</Text>
            </View>
            <View style={styles.headIcon}>
              <Svg width={30} height={30} viewBox="0 0 24 24" fill="none">
                <Path
                  d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"
                  fill="#fff"
                />
              </Svg>
            </View>
          </View>
        ))}
      </ScrollView>
      <View style={styles.micContainer}>
        <TouchableOpacity onPress={handleMicPress} disabled={micActive}>
          {micActive ? (
            <View style={styles.dotsContainer}>
              <Text style={styles.dots}>...</Text>
            </View>
          ) : (
            <Svg width={64} height={64} viewBox="0 0 24 24" fill="none">
              <Line
                x1="12"
                y1="22.5"
                x2="12"
                y2="18.68"
                stroke="#020202"
                strokeWidth={1.91}
                strokeMiterlimit={10}
              />
              <Rect
                x="8.18"
                y="1.5"
                width="7.64"
                height="13.36"
                rx="3.7"
                stroke="#020202"
                strokeWidth={1.91}
                fill="none"
                strokeMiterlimit={10}
              />
              <Path
                d="M19.64,11.05h0A7.64,7.64,0,0,1,12,18.68h0a7.64,7.64,0,0,1-7.64-7.63h0"
                stroke="#020202"
                strokeWidth={1.91}
                fill="none"
                strokeMiterlimit={10}
              />
              <Line
                x1="9.14"
                y1="22.5"
                x2="14.86"
                y2="22.5"
                stroke="#020202"
                strokeWidth={1.91}
                strokeMiterlimit={10}
              />
            </Svg>
          )}
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 40,
    fontWeight: 'bold',
  },
  topBar: {
    alignItems: 'right',
    marginTop: 40,
    marginLeft: 20,
    width: '80%',
  },
  cameraView: {
    height: 387,
    width: 368,
    alignSelf: 'center',
    marginTop: 20,
    borderWidth: 5,
    borderColor: 'black',
    borderRadius: 10,
    overflow: 'hidden',
  },
  camera: {
    flex: 1,
    width: '110%',
    height: '110%',
  },
  chatContainer: {
    height: 210,
    width: '100%',
    borderTopColor: 'black',
    borderTopWidth: 1,
    borderBottomWidth: 1,
    marginTop: 50,
    backgroundColor: '#f8f8f8',
  },
  micContainer: {
    height: 100,
    width: '100%',
    bottom: 0,
    alignContent: 'center',
    alignItems: 'center',
    marginTop: 40,
  },
  dotsContainer: {
    height: 64,
    width: 64,
    justifyContent: 'center',
    alignItems: 'center',
  },
  dots: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#020202',
    textAlign: 'center',
    lineHeight: 64,
  },
  msgRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    alignSelf: 'flex-end',
    marginBottom: 10,
  },
  bubble: {
    backgroundColor: '#356AA9',
    borderRadius: 20,
    paddingVertical: 10,
    paddingHorizontal: 15,
    maxWidth: '50%',
    borderBottomRightRadius: 5,
    marginRight: 5,
  },
  headIcon: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  bubbleText: {
    color: 'white',
    fontSize: 15,
    lineHeight: 20,
  },
  cameraPlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
  },
  placeholderText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
});

export default App;
