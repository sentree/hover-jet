#pragma once

#include "message_header.hh"

namespace jet {

struct Message {
  MessageHeader header;

  MESSAGE(Message, header);
};

}  //  namespace jet
