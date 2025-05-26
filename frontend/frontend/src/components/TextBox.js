import { FaMicrophone } from "react-icons/fa";
import { BiSolidConversation } from "react-icons/bi";
import { HiSpeakerWave, HiSpeakerXMark } from "react-icons/hi2";

const TextBox = (prop) => {
    return (
        <div className="container flex items-center justify-center">
            <button>
                <BiSolidConversation/>
            </button>
            <input name="textTranslate" placeholder="Please enter text" />
            <button>
                <HiSpeakerWave/>
            </button>
        </div>
    );
};

export default TextBox;