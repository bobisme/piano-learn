from itertools import izip, cycle
import numpy
import scipy.io.wavfile
import tensorflow as tf
import midi


class Data(object):
    def __init__(self, data, split_ratio=0.8):
        self.data = data
        self.split_ratio = 0.8

    @property
    def _split_index(self):
        return int(self.data.shape[0] * self.split_ratio)

    @property
    def shape(self):
        return self.data.shape

    @property
    def training(self):
        return self.data[:self._split_index]

    @property
    def testing(self):
        return self.data[self._split_index:]

    @classmethod
    def load(cls, filename):
        return cls(numpy.load(filename))


class DataSet(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    @classmethod
    def load(cls, file_prefix):
        return cls(
            Data.load(file_prefix + '_features.npy'),
            Data.load(file_prefix + '_labels.npy'))


def unpack_complex(tensor, name='unpack_complex'):
    """
    Unpack complex numbers to real and imaginary components.

    Convert [[complex, complex], [complex, complex]]
         to [[real, imag, real, imag], [real, imag, real, imag]]
    """
    tuples = tf.stack([tf.real(tensor), tf.imag(tensor)], 2)
    return tf.reshape(
        tuples, [-1, tuples.shape[1].value * tuples.shape[2].value],
        name=name)


def get_windows(tensor, frame_count, stride=1, name='get_windows'):
    """
    Sliding window function.

    Given [[1, 2, 3], [4, 5, 6], [7, 8, 9]], frame_count=2
      Get [[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]]

    Reshape with tf.reshape(tensor, [-1, frame_count, feature_length])
    """
    shape = tensor.get_shape().as_list()
    feature_size = shape[1]
    patches = tf.extract_image_patches(
        tf.reshape(tensor, [1, -1, feature_size, 1]),
        [1, frame_count, feature_size, 1], [1, stride, feature_size, 1],
        [1, 1, 1, 1], 'SAME')
    return tf.reshape(patches, [-1, frame_count * feature_size], name=name)


def get_2d_windows(tensor, frame_count, stride=1, name='get_2d_windows'):
    """
    Sliding window function.

    Given [[1, 2, 3], [4, 5, 6], [7, 8, 9]], frame_count=2
      Get [[[1, 2, 3],
            [4, 5, 6]],
           [[4, 5, 6],
            [7, 8, 9]]]

    Reshape with tf.reshape(tensor, [-1, frame_count, feature_length])
    """
    feature_size = tensor.get_shape().as_list()[1]
    windows = get_windows(tensor, frame_count, stride=stride, name=name)
    return tf.reshape(windows, [-1, frame_count, feature_size], name=name)


class MTrack(object):
    def __init__(self, midi_obj):
        self.midi_obj = midi_obj

    def __enter__(self):
        self.pattern = midi.Pattern(format=0)
        self.track = midi.Track()
        self.pattern.append(self.track)
        self.midi_obj.pattern = self.pattern
        return self.track

    def __exit__(self, *args):
        self.track.append(midi.EndOfTrackEvent(tick=500))


class MIDI(object):
    BASE_PATH = ''

    def __init__(self, name, debug=False):
        self.name = name
        self.debug = debug

    def build_track(self):
        return MTrack(self)

    @property
    def wav_path(self):
        return self.BASE_PATH+self.name+'.wav'

    @property
    def midi_path(self):
        return self.BASE_PATH+self.name+'.mid'

    def save_midi(self, filename=None):
        if filename is None:
            filename = self.midi_path
        midi.write_midifile(filename, self.pattern)

    def save(self):
        midi.save_midi()
        self.midi_to_wave()

    @staticmethod
    def _midi_to_wave(midi_path, wav_path):
        subprocess.check_call([
            'mrswatson',
            '--midi-file', midi_path,
            '--output', wav_path,
            '--plugin', 'VSUpright_v1_MacVST'])

    def midi_to_wave(self):
        return self._midi_to_wave(self.midi_path, self.wav_path)

    def get_wav(self):
        return thinkdsp.read_wave(self.wav_path)


def wav_to_frames(filename):
    samples_per_frame = 640
    sample_rate, wave = scipy.io.wavfile.read(filename)
    frame_count = int(len(wave) / samples_per_frame)
    spectra = numpy.zeros((frame_count, 321), dtype=numpy.complex128)

    for i in xrange(frame_count):
        wave_offset = int(samples_per_frame * i)
        slice_ = wave[wave_offset:int(wave_offset + samples_per_frame)]
        spectrum = numpy.fft.rfft(slice_)
        # frequency_centers = numpy.fft.rfftfreq(slice_)
        numpy.copyto(spectra[i], spectrum)
    return spectra


def _gen_label_midi_data():
    states = ('on', 'off')
    notes = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    i = 0
    for state in states:
        pitch = midi.C_2
        for octave in xrange(2, 10):
            for note in notes:
                if pitch > midi.C_9:
                    break
                yield (i, {
                    'name': '{}{} {}'.format(note, octave, state),
                    'pitch': pitch,
                    'on': state == 'on',
                    'octave': octave,
                })
                i += 1
                pitch += 1
reverse_notes = dict(_gen_label_midi_data())


def labels_to_messages(labels):
    ticks_per_frame = 17.6
    last_tick = 0

    def ticks_for(i):
        return int(numpy.round(i * ticks_per_frame))

    i = 0
    for frame_labels in labels:
        for class_i, x in enumerate(frame_labels):
            if x > 0:
                note = reverse_notes[class_i]
                if note['on']:
                    class_ = midi.NoteOnEvent
                else:
                    class_ = midi.NoteOffEvent
                ticks = ticks_for(i)
                tick = ticks - last_tick
                last_tick = ticks
                yield class_(tick=tick, pitch=note['pitch'], velocity=64)
        i += 1
    yield midi.EndOfTrackEvent(tick=1000)


def labels_to_midi(labels, name=None):
    m = MIDI(name)
    with m.build_track() as track:
        for event in labels_to_messages(labels):
            track.append(event)
    return m


def frames_to_midi(name, frames, classifier):
    predictions = classifier.predict(frames)
    return labels_to_midi(name, (x['predicted_classes'] for x in predictions))


class TrackData(object):
    FRAME_SIZE = 0.04 # 40ms
    NOTE_COUNT = midi.C_9 - midi.C_2 + 1
    BEATS_PER_MINUTE = 120

    def __init__(self, name, _lazy=False):
        self.name = name
        if not _lazy:
            self.load_files()
            self.set_metadata()
            self.build_data()

    def load_files(self):
        self.pattern = midi.read_midifile(self.name + '.mid')
        self.pattern.make_ticks_abs()
        self.sample_rate, self.wave = scipy.io.wavfile.read(self.name + '.wav')

    def build_data(self):
        self.spectra = numpy.zeros(
            (self.frame_count, self.spectrum_size), dtype=numpy.complex128)
        self.labels = numpy.zeros(
            (self.frame_count, self.NOTE_COUNT * 2), dtype=numpy.bool)

        for index, class_ in self.gen_labels():
            self.labels[index][class_] = True

        for index, spectrum in self.gen_spectra():
            numpy.copyto(self.spectra[index], spectrum)

    def set_metadata(self):
        self.resolution = self.pattern.resolution
        self.ticks_per_frame = self._get_ticks_per_frame()
        self.samples_per_frame = self._get_samples_per_frame()
        self.frame_count = int(numpy.ceil(
            self.pattern[0][-1].tick / self.ticks_per_frame))
        self.spectrum_size = int(self.samples_per_frame / 2 + 1)

    def gen_labels(self):
        for message in self.pattern[0]:
            index = self.get_index_for_tick(message.tick)
            if isinstance(message, midi.NoteOnEvent):
                yield index, message.pitch - midi.C_2
            elif isinstance(message, midi.NoteOffEvent):
                yield index, message.pitch - midi.C_2 + self.NOTE_COUNT

    def gen_spectra(self):
        for i in xrange(self.frame_count):
            wave_offset = int(self.samples_per_frame * i)
            if wave_offset + self.samples_per_frame >= self.wave.size:
                continue
            slice_ = self.wave[wave_offset:int(wave_offset + self.samples_per_frame)]
            spectrum = numpy.fft.rfft(slice_)
            # frequency_centers = numpy.fft.rfftfreq(slice_)
            yield i, spectrum

    def get_index_for_tick(self, tick):
        return int(tick / self.ticks_per_frame)

    def _get_ticks_per_frame(self):
        seconds_per_beat = 60.0 / self.BEATS_PER_MINUTE
        ticks_per_beat = self.resolution
        seconds_per_tick = seconds_per_beat / ticks_per_beat
        return self.FRAME_SIZE / seconds_per_tick

    def _get_samples_per_frame(self):
        return self.FRAME_SIZE * self.sample_rate

    def get_spectrum_frequencies(self):
        return numpy.fft.rfftfreq(
            int(self.samples_per_frame), 1.0 / self.sample_rate)

    def _gen_label_names(self):
        states = ('on', 'off')
        notes = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        for state in states:
            for octave in xrange(2, 10):
                for note in notes:
                    yield '{}{} {}'.format(note, octave, state)

    def gen_describe_data(self):
        freqs = self.get_spectrum_frequencies()
        for i in xrange(self.frame_count):
            print('{}:'.format(i))
            label = self.labels[i]
            events = []
            for label_i, name in enumerate(self._gen_label_names()):
                if label_i >= self.NOTE_COUNT * 2:
                    break
                if label[label_i]:
                    events.append(name)
            print(*events, sep=', ')
            seaborn.plt.plot(freqs[:80], numpy.abs(self.spectra[i][:80]))
            yield events, self.spectra[i]

    def save(self):
        with open(self.name + '_features.npy', 'w') as f:
            numpy.save(f, self.spectra)
        with open(self.name + '_labels.npy', 'w') as f:
            numpy.save(f, self.labels)

def generate_runs():
    MAX_NOTE = midi.C_9
    MIN_NOTE = midi.C_2

    def gen_full_scale(root, intervals):
        """
        Find the lowest note in the scale at or above C2,
        then cycle all the way up to C9.
        """
        interval_cycle = cycle(intervals)
        next_note = root - 24
        while next_note <= MAX_NOTE:
            if next_note >= MIN_NOTE:
                yield next_note
            note = next_note
            next_note = next(interval_cycle) + note

    def run(root, intervals=None, note_count=8, duration=40):
        def gen_duration():
            while True:
                x = numpy.random.random()
                # generate triplets
                if x < 0.05:
                    d = int(duration * 2.0 / 3.0)
                    yield d
                    yield d
                    yield d
                elif 0.05 <= x < 0.10:
                    yield int(duration / 2.0)
                elif 0.10 <= x < 0.15:
                    yield duration * 2
                else:
                    yield duration

        def gen_velocity():
            x = numpy.random.random()
            while True:
                yield int(numpy.round((numpy.sin(x) + 1) / 2 * 7 + 1) * 16) - 1
                x += 0.05 + (numpy.random.random() - 0.5) / 10

        dur = gen_duration()
        vel = gen_velocity()

        # 1 = up, -1 = down
        direction = 1
        if intervals is None:
            intervals = [1]
        full_scale = list(gen_full_scale(root, intervals))
        i = numpy.random.randint(len(full_scale))
        for _ in xrange(note_count):
            pitch = full_scale[i]
            yield midi.NoteOnEvent(tick=0, velocity=next(vel), pitch=pitch)
            yield midi.NoteOffEvent(tick=next(dur), pitch=pitch)
            if numpy.random.random() < 0.1:
                direction *= -1
            i += direction
            if i < 0:
                direction = 1
                i = 1
            if i > len(full_scale) - 1:
                direction = -1
                i = len(full_scale) - 2

    CHROMATIC = [1] * 11
    MAJOR = [2, 2, 1, 2, 2, 2, 1]
    MAJOR_ARP = [4, 3, 5]
    THE_7 = [3, 4, 3, 2]
    MAJ_7 = [4, 3, 3, 2]
    MAJ_MIN_7 = [3, 4, 4, 1]
    MINOR = [2, 1, 2, 2, 1, 2, 2]
    MINOR_HARMONIC = [2, 1, 2, 2, 1, 3, 1]
    MINOR_ARP = [3, 4, 5]
    WHOLE_TONE = [2, 2, 2, 2, 2, 2]
    OCTATONIC = [1, 2, 1, 2, 1, 2, 1, 2]
    DIMINISHED = [3, 3, 3, 3]
    AUGMENTED = [4, 4, 4]
    FIFTHS = [7, 5]
    OCTAVES = [12]
    MINOR_PENT = [3, 2, 2, 3, 2]
    MAJOR_PENT = [2, 2, 3, 2, 3]
    BLUES = [3, 1, 1, 2, 3, 2]

    INTERVAL_SETS = [
        CHROMATIC, MAJOR, MAJOR_ARP, THE_7, MAJ_7,
        MAJ_MIN_7, MINOR, MINOR_HARMONIC, MINOR_ARP,
        WHOLE_TONE, OCTATONIC, DIMINISHED, AUGMENTED,
        FIFTHS, OCTAVES, MAJOR_PENT, MINOR_PENT, BLUES]

    KEYS = ('C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B')
    DURATIONS = (40, 60, 80, 150, 200)
    for key in KEYS:
        root = getattr(midi, key + '_2')
        m = MIDI('runs_in_'+key)
        with m.build_track() as track:
            for interval_set in INTERVAL_SETS:
                for duration in DURATIONS:
                    note_count = 100 * len(interval_set)
                    track.extend(run(
                        root, intervals=interval_set,
                        note_count=note_count, duration=duration))
        m.save_midi()
        print('Finished MIDI for', key.replace('s', '#'))
    print('Done')
