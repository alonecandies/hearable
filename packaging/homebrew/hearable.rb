# Homebrew formula for a third-party tap (e.g. `brew tap alonecandies/hearable`).
#
# `head` builds from git master and works today. For a stable release, push the v0.1.0 tag,
# then fill the `stable` block's url + sha256 (the sha256 of GitHub's auto-generated
# v0.1.0.tar.gz). The sherpa-onnx native libs are provided as a `resource` so the build
# doesn't need network inside Homebrew's sandbox.
class Hearable < Formula
  desc "Real-time on-device captioning + speaker-ID overlay for Deaf/HoH users"
  homepage "https://github.com/alonecandies/hearable"
  license "MIT"
  head "https://github.com/alonecandies/hearable.git", branch: "master"

  stable do
    url "https://github.com/alonecandies/hearable/archive/refs/tags/v0.1.0.tar.gz"
    sha256 "04bbfb609d78a7b8b20e6a75a1ce3d0285f23258904e92e12d9f6da7bb93282b"
  end

  depends_on "rust" => :build

  # Prebuilt sherpa-onnx static libs (+ ONNX Runtime). sherpa-onnx-sys's build.rs normally
  # downloads these, but Homebrew's build sandbox blocks network — so we stage them here and
  # point SHERPA_ONNX_ARCHIVE_DIR at the directory holding the archive.
  on_macos do
    if Hardware::CPU.arm?
      resource "sherpa-libs" do
        url "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.3/sherpa-onnx-v1.13.3-osx-arm64-static-lib.tar.bz2", using: :nounzip
        sha256 "8a524849ea13db3abe667f5f785280b2396dee17856c912e22cb24d0344b9a5a"
      end
    end
    # x86_64 macOS: add the matching osx-x64 archive + its sha256.
  end
  # on_linux: add the linux-x64 (and aarch64) archives + sha256s, same pattern.

  def install
    # Stage the prebuilt sherpa libs if a resource is defined for this platform.
    if (libs = resources.find { |r| r.name == "sherpa-libs" })
      archive_dir = buildpath/"sherpa-archive"
      archive_dir.mkpath
      libs.stage(archive_dir)
      ENV["SHERPA_ONNX_ARCHIVE_DIR"] = archive_dir.to_s
    end

    system "cargo", "build", "--release", "--features", "live"
    bin.install "target/release/hearable"
  end

  def caveats
    <<~EOS
      hearable needs microphone access and ML model files at runtime.
      Fetch models, then run:
          hearable run --models /path/to/models                 # SenseVoice (zh/en/ja/ko/yue)
          hearable run --models /path/to/models --engine whisper --language vi   # Vietnamese
      On macOS, for the cleanest microphone-permission experience build the .app bundle
      (scripts/bundle-macos.sh) instead of running the bare binary.
    EOS
  end

  test do
    assert_match "hearable", shell_output("#{bin}/hearable --help")
  end
end
